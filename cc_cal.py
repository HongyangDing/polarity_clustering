import itertools
import os
import time

import numpy as np
import torch
from mpi4py import MPI

import config
from lib.corr import correlate_conv1d_matrix
from lib.cut import read_fsta


# =========================================================
# Shelly (2016) 对齐说明
# 在当前工程里, 由于没有直接沿用 matched-filter/template matching
# 所产生的模板-检测事件对应关系, 无法机械地完全复现 Shelly (2016)
# 只在模板及其检测事件之间计算测量的那一层工作流。
#
# 因此当前默认实现采用 Shelly-proxy 模式:
# 1. 使用 signed correlation coefficient 作为相对极性测量
# 2. 对低于阈值的 |CC| 直接置零
# 3. 使用 SVD 的第一左奇异向量构造事件特征
#
# 其中第 2 步是代理约束:
# 用低 CC 置零去近似 "只有模板及其相关 detected event 才有有效测量"。
#
# 下列参数/模式不是 Shelly (2016) 文中直接给出的固定参数:
# - SHELLY2016_PROXY_ABS_CC_MIN = 0.5
# - LEGACY_PEAK_GAP_CC_MIN = 0.5
# - LEGACY_PEAK_GAP_WEIGHT_MIN = 1e-5
# - RELATIVE_POLARITY_MODE = "legacy_peak_gap"
# - SVD_MODE = "lowrank"
# =========================================================
RELATIVE_POLARITY_MODE = os.environ.get(
    "CC_RELATIVE_POLARITY_MODE",
    "shelly2016_proxy",
)

SHELLY2016_SIGNED_CC_ABS_MIN = None
SHELLY2016_PROXY_ABS_CC_MIN = 0.5

LEGACY_PEAK_GAP_CC_MIN = 0.5
LEGACY_PEAK_GAP_WEIGHT_MIN = 1e-5
TEMPLATE_BATCH_SIZE = int(os.environ.get("CC_TEMPLATE_BATCH_SIZE", "4"))
SVD_MODE = os.environ.get("CC_SVD_MODE", "reduced")  # "reduced" or "lowrank"
LOWRANK_NITER = int(os.environ.get("CC_SVD_NITER", "2"))


def slice_correlation_lag_window(corr_mat, pha_id, fs):
    pha_id = int(pha_id)
    lag_max_sec = 0.5 if pha_id == 0 else 0.8
    lag_max_samp = int(round(lag_max_sec * fs))

    nlag = corr_mat.shape[-1]
    zero_idx = (nlag - 1) // 2
    left = max(0, zero_idx - lag_max_samp)
    right = min(nlag, zero_idx + lag_max_samp + 1)

    corr_use = corr_mat[..., left:right]
    abs_use = torch.abs(corr_use)
    return corr_use, abs_use


def get_peak_signed_correlation(corr_mat, pha_id, fs):
    corr_use, abs_use = slice_correlation_lag_window(corr_mat, pha_id, fs)

    idx1 = torch.argmax(abs_use, dim=-1, keepdim=True)
    cc1 = torch.gather(corr_use, -1, idx1).squeeze(-1)
    abs1 = torch.gather(abs_use, -1, idx1).squeeze(-1)
    return cc1, abs1


def get_weighted_relative_polarity(corr_mat, pha_id, fs, min_sep_sec=0.03):
    corr_use, abs_use = slice_correlation_lag_window(corr_mat, pha_id, fs)
    idx1 = torch.argmax(abs_use, dim=-1, keepdim=True)
    cc1 = torch.gather(corr_use, -1, idx1).squeeze(-1)
    abs1 = torch.gather(abs_use, -1, idx1).squeeze(-1)
    min_sep_samp = max(1, int(round(min_sep_sec * fs)))

    idx_grid = torch.arange(corr_use.shape[-1], device=corr_use.device)
    idx_grid = idx_grid.view(*([1] * (corr_use.ndim - 1)), -1)
    mask_secondary = torch.abs(idx_grid - idx1) >= min_sep_samp

    abs_use_2 = abs_use.clone()
    abs_use_2[~mask_secondary] = -1.0

    idx2 = torch.argmax(abs_use_2, dim=-1, keepdim=True)
    abs2 = torch.gather(abs_use, -1, idx2).squeeze(-1)

    r = torch.sign(cc1) * (abs1 - abs2)
    return r, abs1, abs2


def apply_pair_validity_mask(r, abs1, abs2, cc_min=None, weight_min=None):
    valid = torch.ones_like(r, dtype=torch.bool)
    if cc_min is not None:
        valid &= abs1 >= cc_min
    if weight_min is not None:
        valid &= (abs1 - abs2) >= weight_min
    return torch.where(valid, r, torch.zeros_like(r))


def choose_device(rank):
    if torch.cuda.is_available():
        local_rank = int(
            os.environ.get(
                "SLURM_LOCALID",
                os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", rank),
            )
        )
        return torch.device(f"cuda:{local_rank % torch.cuda.device_count()}")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_job_list():
    sta_dict = read_fsta(config.fsta)
    return list(itertools.product(sta_dict.keys(), config.channel_list, config.phase_ids))


def first_left_singular_vector(r_k):
    if not torch.any(r_k != 0).item():
        return torch.zeros(r_k.shape[0], device=r_k.device)

    if r_k.shape[1] == 1:
        col = r_k[:, 0]
        norm = torch.linalg.norm(col)
        if norm <= 0:
            return torch.zeros_like(col)
        return col / norm

    if SVD_MODE == "reduced":
        u, _, _ = torch.linalg.svd(r_k, full_matrices=False)
        return u[:, 0]

    if SVD_MODE == "lowrank":
        q = min(2, min(r_k.shape))
        u, _, _ = torch.svd_lowrank(r_k, q=q, niter=LOWRANK_NITER)
        return u[:, 0]

    raise ValueError("CC_SVD_MODE must be 'reduced' or 'lowrank'.")


def build_feature_column(temp_mat, detected_mat, pha_id, fs):
    r_k = torch.zeros(detected_mat.shape[0], temp_mat.shape[0], device=detected_mat.device)
    batch_size = max(1, int(TEMPLATE_BATCH_SIZE))

    for start in range(0, temp_mat.shape[0], batch_size):
        stop = min(start + batch_size, temp_mat.shape[0])
        temp_batch = temp_mat[start:stop, :]
        corr_mat = correlate_conv1d_matrix(temp_batch, detected_mat)

        if RELATIVE_POLARITY_MODE == "shelly2016_signed_cc":
            cc1, abs1 = get_peak_signed_correlation(
                corr_mat=corr_mat,
                pha_id=pha_id,
                fs=fs,
            )
            if SHELLY2016_SIGNED_CC_ABS_MIN is None:
                r = cc1
            else:
                r = torch.where(
                    abs1 >= SHELLY2016_SIGNED_CC_ABS_MIN,
                    cc1,
                    torch.zeros_like(cc1),
                )
        elif RELATIVE_POLARITY_MODE == "shelly2016_proxy":
            cc1, abs1 = get_peak_signed_correlation(
                corr_mat=corr_mat,
                pha_id=pha_id,
                fs=fs,
            )
            r = torch.where(
                abs1 >= SHELLY2016_PROXY_ABS_CC_MIN,
                cc1,
                torch.zeros_like(cc1),
            )
        elif RELATIVE_POLARITY_MODE == "legacy_peak_gap":
            r, abs1, abs2 = get_weighted_relative_polarity(
                corr_mat=corr_mat,
                pha_id=pha_id,
                fs=fs,
                min_sep_sec=0.03,
            )
            r = apply_pair_validity_mask(
                r=r,
                abs1=abs1,
                abs2=abs2,
                cc_min=LEGACY_PEAK_GAP_CC_MIN,
                weight_min=LEGACY_PEAK_GAP_WEIGHT_MIN,
            )
        else:
            raise ValueError(
                "CC_RELATIVE_POLARITY_MODE must be "
                "'shelly2016_proxy', 'shelly2016_signed_cc', or 'legacy_peak_gap'."
            )

        r_k[:, start:stop] = r.transpose(0, 1)

    active_rows = torch.any(r_k != 0, dim=1)
    active_cols = torch.any(r_k != 0, dim=0)
    if (not active_rows.any().item()) or (not active_cols.any().item()):
        return torch.zeros(r_k.shape[0], device=r_k.device)

    r_active = r_k[active_rows][:, active_cols]
    leading_active = first_left_singular_vector(r_active)
    leading = torch.zeros(r_k.shape[0], device=r_k.device)
    leading[active_rows] = leading_active
    return leading


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        t0 = time.time()
        sta_pha = load_job_list()
        sta_pha_order = {key: idx for idx, key in enumerate(sta_pha)}
        sta_pha_div = np.array_split(np.array(sta_pha, dtype=object), size)
    else:
        sta_pha_order = None
        sta_pha_div = None

    sta_pha_order = comm.bcast(sta_pha_order, root=0)
    sta_pha_div = comm.bcast(sta_pha_div, root=0)

    jobs_for_rank = [tuple(job) for job in sta_pha_div[rank].tolist()]
    device = choose_device(rank)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
    combined_r = None

    for count, (sta, channel_code, pha_id) in enumerate(jobs_for_rank, start=1):
        if count % 10 == 1:
            print(f"rank:{rank}, device:{device}, {count}/{len(jobs_for_rank)}")

        mat_name = f"{sta}_{channel_code}_{pha_id}.npy"
        detected_mat_path = os.path.join(config.output_detected, mat_name)
        temp_mat_path = os.path.join(config.output_temp, mat_name)

        try:
            detected_mat = np.load(detected_mat_path)
            temp_mat = np.load(temp_mat_path)
        except Exception:
            print(f"[WARN] missing waveform matrix: {sta} {channel_code} {pha_id}")
            continue

        if combined_r is None:
            combined_r = torch.zeros(
                detected_mat.shape[0],
                len(sta_pha_order),
                device=device,
            )

        detected_valid = np.any(detected_mat != 0, axis=1)
        temp_valid = np.any(temp_mat != 0, axis=1)
        if (not detected_valid.any()) or (not temp_valid.any()):
            continue

        detected_mask = torch.from_numpy(detected_valid).to(device=device)
        detected_sel = torch.tensor(
            detected_mat[detected_valid],
            dtype=torch.float32,
            device=device,
        )
        temp_sel = torch.tensor(
            temp_mat[temp_valid],
            dtype=torch.float32,
            device=device,
        )
        col_sel = build_feature_column(
            temp_mat=temp_sel,
            detected_mat=detected_sel,
            pha_id=int(pha_id),
            fs=config.sampling_rate,
        )
        combined_r[detected_mask, sta_pha_order[(sta, channel_code, pha_id)]] = col_sel
        del detected_sel, temp_sel, col_sel, detected_mask

    combined_r_cpu = None if combined_r is None else combined_r.to("cpu").numpy()
    combined_r_gather = comm.gather(combined_r_cpu, root=0)
    if rank != 0:
        return

    valid_chunks = [chunk for chunk in combined_r_gather if chunk is not None]
    if not valid_chunks:
        raise RuntimeError("No valid waveform pairs were found; combined feature matrix is empty.")

    combined_rr = np.zeros_like(valid_chunks[0], dtype=np.float32)
    for chunk in valid_chunks:
        combined_rr += chunk.astype(np.float32, copy=False)

    np.save("combined_RR_all_2col.npy", combined_rr)
    print(f"total time: {time.time() - t0}")
    print(f"[INFO] Saved combined feature matrix: combined_RR_all_2col.npy")
    print(f"[INFO] Relative polarity mode     : {RELATIVE_POLARITY_MODE}")
    if RELATIVE_POLARITY_MODE == "shelly2016_proxy":
        print(f"[INFO] Shelly proxy |CC| min    : {SHELLY2016_PROXY_ABS_CC_MIN}")
    elif RELATIVE_POLARITY_MODE == "shelly2016_signed_cc":
        print(f"[INFO] Strict Shelly |CC| min   : {SHELLY2016_SIGNED_CC_ABS_MIN}")
    else:
        print(f"[INFO] Legacy peak-gap cc min   : {LEGACY_PEAK_GAP_CC_MIN}")
        print(f"[INFO] Legacy peak-gap w min    : {LEGACY_PEAK_GAP_WEIGHT_MIN}")
    print(f"[INFO] CC template batch size    : {TEMPLATE_BATCH_SIZE}")
    print(f"[INFO] CC SVD mode               : {SVD_MODE}")


if __name__ == "__main__":
    main()
