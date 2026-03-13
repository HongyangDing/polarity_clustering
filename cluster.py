import os
from datetime import datetime, timezone

import config

if getattr(config, "step3_mplconfig_dir", None):
    mplconfig_dir = os.path.abspath(config.step3_mplconfig_dir)
    os.makedirs(mplconfig_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", mplconfig_dir)

import matplotlib
import numpy as np
from matplotlib.colors import to_hex

if getattr(config, "step3_matplotlib_backend", None):
    matplotlib.use(config.step3_matplotlib_backend)
else:
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        matplotlib.use("TkAgg")
    else:
        matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

from lib.cut import read_fpha_list


# =========================================================
# 聚类主配置
# 这份脚本默认尽量贴近 Shelly (2016) 的主思路：
# 事件极性向量 + cosine distance + hierarchical clustering
#
# 下列参数属于实现选择或展示设置, 不是 Shelly (2016) 文中明确给出的固定参数:
# - config.step3_linkage_method
# - config.step3_cluster_cut_mode
# - config.step3_distance_cut_threshold
# - config.step3_maxclust_n_clusters
# - config.step3_top_n_show / config.step3_show_noise / config.step3_export_html
#
# 默认参数统一放在 config.py 的 Step 3 区域。
# =========================================================
FDETECTED = config.fdetected

# ---------- 公共参数 ----------
LINKAGE_METHOD = config.step3_linkage_method
CLUSTER_CUT_MODE = config.step3_cluster_cut_mode  # "distance" or "maxclust"

DISTANCE_THRESHOLD = config.step3_distance_cut_threshold
MAXCLUST_N_CLUSTERS = config.step3_maxclust_n_clusters

# ---------- 绘图与输出 ----------
TOP_N_SHOW = config.step3_top_n_show
SHOW_NOISE = config.step3_show_noise
SAVE_FIG = config.step3_save_fig
FIG_OUT = config.step3_fig_out
POINT_SIZE = config.step3_point_size
BG_POINT_SIZE = config.step3_bg_point_size

# ---------- HTML 时间滑条输出 ----------
EXPORT_HTML = config.step3_export_html
HTML_OUT = config.step3_html_out
HTML_MARKER_SIZE = config.step3_html_marker_size
HTML_SLIDER_NFRAMES = config.step3_html_slider_nframes
OTHER_CLUSTER_COLOR = config.step3_other_cluster_color
MAG_HIGHLIGHT = config.step3_mag_highlight
MAG_HIGHLIGHT_MULT = config.step3_mag_highlight_mult
FUTURE_ALPHA = config.step3_future_alpha
PAST_ALPHA = config.step3_past_alpha


def get_cut_plan(n_valid):
    if CLUSTER_CUT_MODE == "distance":
        if DISTANCE_THRESHOLD <= 0:
            raise ValueError("DISTANCE_THRESHOLD must be > 0.")
        return {
            "criterion": "distance",
            "t": DISTANCE_THRESHOLD,
            "label": "distance",
            "value_text": f"{DISTANCE_THRESHOLD}",
        }

    if CLUSTER_CUT_MODE == "maxclust":
        if MAXCLUST_N_CLUSTERS <= 0:
            raise ValueError("MAXCLUST_N_CLUSTERS must be > 0.")
        n_clusters_eff = min(int(MAXCLUST_N_CLUSTERS), int(n_valid))
        return {
            "criterion": "maxclust",
            "t": n_clusters_eff,
            "label": "maxclust",
            "value_text": f"{MAXCLUST_N_CLUSTERS} (effective={n_clusters_eff})",
        }

    raise ValueError("CLUSTER_CUT_MODE must be 'distance' or 'maxclust'.")


def load_feature_matrix(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature matrix not found: {path}")
    arr = np.load(path)
    print(f"[INFO] Loaded feature matrix from: {path}, shape={arr.shape}")
    return arr


def sanitize_feature_matrix(X):
    X = np.asarray(X, dtype=float)

    finite_mask = np.all(np.isfinite(X), axis=1)
    row_norm = np.linalg.norm(X, axis=1)
    nonzero_mask = row_norm > 0
    valid_mask = finite_mask & nonzero_mask

    X_valid = X[valid_mask].copy()
    return X_valid, valid_mask

def relabel_by_size(labels):
    labels = np.asarray(labels, dtype=int)
    nonzero = labels[labels > 0]
    if nonzero.size == 0:
        return labels.copy(), {}

    unique, counts = np.unique(nonzero, return_counts=True)
    ranked = sorted(zip(unique, counts), key=lambda x: x[1], reverse=True)
    mapping = {old: new for new, (old, _) in enumerate(ranked, start=1)}

    out = np.zeros_like(labels)
    for old, new in mapping.items():
        out[labels == old] = new
    return out, mapping


def build_cluster_order(labels):
    cluster_to_idx = {}
    for cid in np.unique(labels):
        if cid == 0:
            continue
        idx = np.where(labels == cid)[0]
        cluster_to_idx[cid] = idx

    cluster_sizes = {cid: len(idx) for cid, idx in cluster_to_idx.items()}
    ranked = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)
    cluster_ids = [cid for cid, _ in ranked]
    return cluster_ids, cluster_to_idx, cluster_sizes


def cosine_distance(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return np.nan
    return 1.0 - np.dot(a, b) / (na * nb)


def cluster_medoid_info(X_proc_full, labels, all_event_ids, cluster_ids):
    medoid_vecs = {}
    medoid_eids = {}
    medoid_indices = {}

    for cid in cluster_ids:
        idx = np.where(labels == cid)[0]
        Xc = X_proc_full[idx]
        if Xc.shape[0] == 0:
            continue

        score = Xc @ Xc.sum(axis=0)
        j = int(np.argmax(score))
        gidx = idx[j]

        medoid_vecs[cid] = X_proc_full[gidx]
        medoid_eids[cid] = str(all_event_ids[gidx])
        medoid_indices[cid] = gidx

    return medoid_vecs, medoid_eids, medoid_indices


def hierarchical_cluster_features(X):
    X_valid_raw, valid_mask = sanitize_feature_matrix(X)
    if X_valid_raw.shape[0] == 0:
        raise ValueError("No valid rows remain after removing NaN/Inf/zero rows.")

    X_valid = X_valid_raw

    labels_full = np.zeros(X.shape[0], dtype=int)
    X_proc_full = np.zeros_like(X, dtype=float)
    X_proc_full[valid_mask] = X_valid

    if X_valid.shape[0] == 1:
        labels_full[valid_mask] = 1
        cut_plan = get_cut_plan(1)
        return labels_full, valid_mask, X_proc_full, cut_plan

    D = pdist(X_valid, metric="cosine")
    Z = linkage(D, method=LINKAGE_METHOD)
    cut_plan = get_cut_plan(X_valid.shape[0])
    labels_valid = fcluster(Z, t=cut_plan["t"], criterion=cut_plan["criterion"])

    labels_full[valid_mask] = labels_valid
    labels_full, _ = relabel_by_size(labels_full)
    return labels_full, valid_mask, X_proc_full, cut_plan


def save_cluster_summary(out_file, cluster_ids, cluster_sizes, medoid_eids):
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("# rank  cluster_id  size  medoid_event_id\n")
        for rank, cid in enumerate(cluster_ids, start=1):
            f.write(
                f"{rank:4d} {cid:10d} {cluster_sizes[cid]:6d} {medoid_eids.get(cid, '-1')}\n"
            )


def save_cluster_distance_matrix(out_file, shown_cluster_ids, medoid_vecs):
    with open(out_file, "w", encoding="utf-8") as f:
        if len(shown_cluster_ids) == 0:
            f.write("# no clusters\n")
            return

        f.write("# cosine distance matrix between shown clusters (medoid vectors)\n")
        header = "cluster" + "".join([f" {cid:>10d}" for cid in shown_cluster_ids])
        f.write(header + "\n")

        for c1 in shown_cluster_ids:
            row = [f"{c1:>7d}"]
            for c2 in shown_cluster_ids:
                d = cosine_distance(medoid_vecs[c1], medoid_vecs[c2])
                row.append(f" {d:10.6f}")
            f.write("".join(row) + "\n")


def get_event_xyz_from_dic(all_dic, event_ids):
    lats, lons, deps = [], [], []
    for eid in event_ids:
        ot, lat, lon, dep, mag, evid = all_dic[str(eid)][0]
        lats.append(lat)
        lons.append(lon)
        deps.append(dep)
    return np.array(lats), np.array(lons), np.array(deps)


def collect_event_arrays(all_dic, event_ids):
    dt_utc, lat, lon, dep, mag = [], [], [], [], []
    for eid in event_ids:
        ot, la, lo, de, ma, evid = all_dic[str(eid)][0]
        dt_utc.append(ot)
        lat.append(la)
        lon.append(lo)
        dep.append(de)
        mag.append(ma)

    return (
        np.array(dt_utc, dtype=object),
        np.array(lat, dtype=float),
        np.array(lon, dtype=float),
        np.array(dep, dtype=float),
        np.array(mag, dtype=float),
    )


def lonlat_to_local_km(lon, lat):
    lon0 = float(np.mean(lon))
    lat0 = float(np.mean(lat))
    lat0_rad = np.deg2rad(lat0)

    km_per_deg_lon = 111.32 * np.cos(lat0_rad)
    km_per_deg_lat = 110.57

    x_km = (lon - lon0) * km_per_deg_lon
    y_km = (lat - lat0) * km_per_deg_lat
    return x_km, y_km, lon0, lat0


def build_cluster_palette(cluster_ids):
    if len(cluster_ids) == 0:
        return {}
    cmap = plt.get_cmap("tab10")
    return {int(cid): to_hex(cmap(i % 10)) for i, cid in enumerate(cluster_ids)}


def _mag_scale(mag):
    m = np.asarray(mag, dtype=float)

    m_lo = np.nanpercentile(m, 10)
    m_hi = np.nanpercentile(m, 90)
    if (not np.isfinite(m_lo)) or (not np.isfinite(m_hi)) or (m_hi <= m_lo):
        m_lo = np.nanmin(m)
        m_hi = np.nanmax(m)

    denom = max(float(m_hi - m_lo), 1e-6)
    u = (m - m_lo) / denom
    u = np.clip(u, 0.0, 1.0)

    scale = 0.7 + 1.1 * u
    scale = np.where(m >= MAG_HIGHLIGHT, scale * MAG_HIGHLIGHT_MULT, scale)
    return np.clip(scale, 0.2, 10.0)


def mag_to_plotly_size(mag, base_size):
    return float(base_size) * _mag_scale(mag)


def export_html_plotly_slider_by_cluster(
    x_km,
    y_km,
    z_km,
    dt_utc,
    mag,
    labels,
    html_path,
    title,
    top_clusters,
    palette,
    marker_size,
    nframes,
    future_alpha,
    past_alpha,
):
    try:
        import plotly.graph_objects as go
    except ImportError as e:
        raise RuntimeError("缺少 plotly，无法导出 HTML。") from e

    dt_py = np.array(
        [ot.datetime.replace(tzinfo=timezone.utc) for ot in dt_utc],
        dtype=object,
    )
    t_ms = np.array([int(d.timestamp() * 1000) for d in dt_py], dtype=np.int64)
    if len(t_ms) == 0:
        raise RuntimeError("没有可导出的事件，无法写 HTML。")

    order = np.argsort(t_ms)
    t_ms = t_ms[order]
    dt_py = dt_py[order]
    x = np.asarray(x_km)[order]
    y = np.asarray(y_km)[order]
    z = np.asarray(z_km)[order]
    labels = np.asarray(labels)[order]
    mag = np.asarray(mag)[order]
    msz = np.asarray(marker_size, dtype=float)[order]

    top_set = set(int(cid) for cid in top_clusters)
    color_arr = []
    cluster_text = []
    for cid in labels:
        cid_int = int(cid)
        if cid_int in top_set:
            color_arr.append(palette[cid_int])
            cluster_text.append(f"C{cid_int}")
        else:
            color_arr.append(OTHER_CLUSTER_COLOR)
            cluster_text.append("other")

    hover_text = [
        f"{dt_py[i].strftime('%Y-%m-%d %H:%M:%S UTC')}<br>{cluster_text[i]}<br>M={mag[i]:.2f}"
        for i in range(len(dt_py))
    ]

    tmin = int(t_ms.min())
    tmax = int(t_ms.max())
    if tmax <= tmin:
        tmax = tmin + 1

    nframes = int(max(nframes, 2))
    cuts = np.linspace(tmin, tmax, nframes).astype(np.int64)

    def split_index(cut):
        return int(np.searchsorted(t_ms, cut, side="right"))

    k0 = split_index(cuts[0])

    legend_traces = []
    for cid in top_clusters:
        legend_traces.append(
            go.Scatter3d(
                x=[None],
                y=[None],
                z=[None],
                mode="markers",
                marker=dict(size=6, color=palette[int(cid)]),
                name=f"C{int(cid)}",
                showlegend=True,
                hoverinfo="skip",
            )
        )
    legend_traces.append(
        go.Scatter3d(
            x=[None],
            y=[None],
            z=[None],
            mode="markers",
            marker=dict(size=6, color=OTHER_CLUSTER_COLOR),
            name="other",
            showlegend=True,
            hoverinfo="skip",
        )
    )

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x[k0:],
                y=y[k0:],
                z=z[k0:],
                mode="markers",
                marker=dict(
                    size=msz[k0:],
                    color=f"rgba(160,160,160,{future_alpha})",
                    line=dict(width=0),
                ),
                hoverinfo="skip",
                showlegend=False,
                name="future",
            ),
            go.Scatter3d(
                x=x[:k0],
                y=y[:k0],
                z=z[:k0],
                mode="markers",
                text=hover_text[:k0],
                hovertemplate="E=%{x:.2f} km<br>N=%{y:.2f} km<br>Depth=%{z:.2f} km<br>%{text}<extra></extra>",
                marker=dict(
                    size=msz[:k0],
                    color=color_arr[:k0],
                    opacity=past_alpha,
                    line=dict(width=0),
                ),
                showlegend=False,
                name="past",
            ),
        ]
        + legend_traces,
        layout=go.Layout(
            title=f"{title} | up to {datetime.fromtimestamp(cuts[0] / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            scene=dict(
                xaxis_title="East (km)",
                yaxis_title="North (km)",
                zaxis_title="Depth (km, down)",
            ),
            margin=dict(l=0, r=0, b=0, t=50),
            legend=dict(x=0.02, y=0.98),
        ),
        frames=[],
    )

    frames = []
    for i, cut in enumerate(cuts):
        k = split_index(cut)
        frames.append(
            go.Frame(
                name=str(i),
                data=[
                    go.Scatter3d(
                        x=x[k:],
                        y=y[k:],
                        z=z[k:],
                        mode="markers",
                        marker=dict(
                            size=msz[k:],
                            color=f"rgba(160,160,160,{future_alpha})",
                            line=dict(width=0),
                        ),
                        hoverinfo="skip",
                        showlegend=False,
                    ),
                    go.Scatter3d(
                        x=x[:k],
                        y=y[:k],
                        z=z[:k],
                        mode="markers",
                        text=hover_text[:k],
                        hovertemplate="E=%{x:.2f} km<br>N=%{y:.2f} km<br>Depth=%{z:.2f} km<br>%{text}<extra></extra>",
                        marker=dict(
                            size=msz[:k],
                            color=color_arr[:k],
                            opacity=past_alpha,
                            line=dict(width=0),
                        ),
                        showlegend=False,
                    ),
                ],
                traces=[0, 1],
                layout=go.Layout(
                    title=f"{title} | up to {datetime.fromtimestamp(cut / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"
                ),
            )
        )

    fig.frames = frames
    fig.update_layout(
        sliders=[
            dict(
                steps=[
                    dict(
                        method="animate",
                        args=[
                            [str(i)],
                            dict(
                                mode="immediate",
                                frame=dict(duration=0, redraw=True),
                                transition=dict(duration=0),
                            ),
                        ],
                        label=datetime.fromtimestamp(cuts[i] / 1000, tz=timezone.utc).strftime("%Y-%m-%d"),
                    )
                    for i in range(len(cuts))
                ],
                currentvalue=dict(prefix="Time: "),
                pad=dict(t=30),
            )
        ]
    )

    fig.write_html(html_path, include_plotlyjs="cdn")
    print(f"[OK] Wrote HTML: {html_path}")


def plot_map_and_profiles(
    shown_cluster_ids,
    cluster_to_idx,
    all_dic,
    all_event_ids,
    labels,
    show_noise=False,
    fig_out=None,
    save_fig=True,
):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    ax_map, ax_ew, ax_ns = axes

    all_lat, all_lon, all_dep = get_event_xyz_from_dic(all_dic, all_event_ids)

    ax_map.scatter(
        all_lon,
        all_lat,
        s=BG_POINT_SIZE,
        c="0.85",
        alpha=0.6,
        edgecolors="none",
        label="All events",
    )
    ax_ew.scatter(
        all_lon,
        all_dep,
        s=BG_POINT_SIZE,
        c="0.85",
        alpha=0.6,
        edgecolors="none",
    )
    ax_ns.scatter(
        all_lat,
        all_dep,
        s=BG_POINT_SIZE,
        c="0.85",
        alpha=0.6,
        edgecolors="none",
    )

    if show_noise:
        noise_idx = np.where(labels == 0)[0]
        if len(noise_idx) > 0:
            noise_eids = all_event_ids[noise_idx]
            nlat, nlon, ndep = get_event_xyz_from_dic(all_dic, noise_eids)

            ax_map.scatter(
                nlon,
                nlat,
                s=POINT_SIZE,
                c="k",
                alpha=0.25,
                edgecolors="none",
                label="Noise/invalid",
            )
            ax_ew.scatter(
                nlon,
                ndep,
                s=POINT_SIZE,
                c="k",
                alpha=0.25,
                edgecolors="none",
            )
            ax_ns.scatter(
                nlat,
                ndep,
                s=POINT_SIZE,
                c="k",
                alpha=0.25,
                edgecolors="none",
            )

    cmap = plt.get_cmap("tab10")

    for i, cid in enumerate(shown_cluster_ids):
        idx = cluster_to_idx[cid]
        eids = all_event_ids[idx]
        lat, lon, dep = get_event_xyz_from_dic(all_dic, eids)
        color = cmap(i % 10)

        ax_map.scatter(
            lon,
            lat,
            s=POINT_SIZE,
            color=color,
            alpha=0.95,
            edgecolors="none",
            label=f"C{cid} (n={len(eids)})",
        )
        ax_ew.scatter(
            lon,
            dep,
            s=POINT_SIZE,
            color=color,
            alpha=0.95,
            edgecolors="none",
        )
        ax_ns.scatter(
            lat,
            dep,
            s=POINT_SIZE,
            color=color,
            alpha=0.95,
            edgecolors="none",
        )

    ax_map.set_xlabel("Longitude")
    ax_map.set_ylabel("Latitude")
    ax_map.set_title("Cluster map")

    ax_ew.set_xlabel("Longitude")
    ax_ew.set_ylabel("Depth (km)")
    ax_ew.set_title("E-W depth profile")
    ax_ew.invert_yaxis()

    ax_ns.set_xlabel("Latitude")
    ax_ns.set_ylabel("Depth (km)")
    ax_ns.set_title("N-S depth profile")
    ax_ns.invert_yaxis()

    for ax in axes:
        ax.grid(alpha=0.25)

    ax_map.legend(fontsize=8, loc="best")
    plt.tight_layout()

    if save_fig and fig_out is not None:
        plt.savefig(fig_out, dpi=220)
        print(f"[INFO] Saved figure: {fig_out}")

    if plt.get_backend().lower().endswith("agg"):
        plt.close(fig)
    else:
        plt.show()


def main():
    detected_dic = read_fpha_list(FDETECTED)
    detected_ids = np.array(list(detected_dic.keys()), dtype=object)

    used_fn = config.step2_feature_matrix_out
    combined_RR = load_feature_matrix(used_fn)

    if combined_RR.shape[0] != len(detected_ids):
        raise ValueError(
            f"Feature matrix row count ({combined_RR.shape[0]}) does not match "
            f"event count ({len(detected_ids)})."
        )

    labels, valid_mask, X_proc_full, cut_plan = hierarchical_cluster_features(combined_RR)

    cluster_ids, cluster_to_idx, cluster_sizes = build_cluster_order(labels)
    if len(cluster_ids) == 0:
        print("[WARN] No nonzero clusters found.")
        return

    shown_cluster_ids = cluster_ids[:TOP_N_SHOW]
    medoid_vecs, medoid_eids, _ = cluster_medoid_info(
        X_proc_full, labels, detected_ids, cluster_ids
    )

    np.save("cluster_labels.npy", labels)
    np.save("cluster_valid_mask.npy", valid_mask)
    save_cluster_summary(
        "cluster_summary.txt",
        cluster_ids,
        cluster_sizes,
        medoid_eids,
    )
    save_cluster_distance_matrix(
        "cluster_distance_matrix.txt",
        shown_cluster_ids,
        medoid_vecs,
    )

    if EXPORT_HTML:
        dt_utc, lat, lon, dep, mag = collect_event_arrays(detected_dic, detected_ids)
        x_km, y_km, lon0, lat0 = lonlat_to_local_km(lon, lat)
        z_km = -dep
        palette = build_cluster_palette(shown_cluster_ids)
        html_title = (
            f"Detected events by cluster | "
            f"lon0={lon0:.4f}, lat0={lat0:.4f} | N={len(labels)}"
        )
        html_size = mag_to_plotly_size(mag, HTML_MARKER_SIZE)
        export_html_plotly_slider_by_cluster(
            x_km=x_km,
            y_km=y_km,
            z_km=z_km,
            dt_utc=dt_utc,
            mag=mag,
            labels=labels,
            html_path=HTML_OUT,
            title=html_title,
            top_clusters=shown_cluster_ids,
            palette=palette,
            marker_size=html_size,
            nframes=HTML_SLIDER_NFRAMES,
            future_alpha=FUTURE_ALPHA,
            past_alpha=PAST_ALPHA,
        )

    print("=" * 60)
    print("[INFO] Algorithm               : hierarchical_cosine")
    print(f"[INFO] Total events            : {len(detected_ids)}")
    print(f"[INFO] Valid rows for cluster  : {np.sum(valid_mask)}")
    print(f"[INFO] Invalid rows            : {np.sum(~valid_mask)}")
    print(f"[INFO] Feature matrix used     : {used_fn}")
    print(f"[INFO] Linkage method          : {LINKAGE_METHOD}")
    print(f"[INFO] Cluster cut mode        : {cut_plan['label']}")
    if cut_plan["label"] == "distance":
        print(f"[INFO] Distance threshold      : {cut_plan['value_text']}")
    else:
        print(f"[INFO] Target cluster count    : {cut_plan['value_text']}")
    print(f"[INFO] Number of kept clusters : {len(cluster_ids)}")
    print(f"[INFO] Top clusters shown      : {shown_cluster_ids}")
    print(f"[INFO] Noise/invalid count     : {np.sum(labels == 0)}")
    print(f"[INFO] Saved figure            : {FIG_OUT}")
    if EXPORT_HTML:
        print(f"[INFO] Saved HTML              : {HTML_OUT}")
    print("[INFO] Saved labels            : cluster_labels.npy")
    print("[INFO] Saved summary           : cluster_summary.txt")
    print("=" * 60)

    plot_map_and_profiles(
        shown_cluster_ids=shown_cluster_ids,
        cluster_to_idx=cluster_to_idx,
        all_dic=detected_dic,
        all_event_ids=detected_ids,
        labels=labels,
        show_noise=SHOW_NOISE,
        fig_out=FIG_OUT,
        save_fig=SAVE_FIG,
    )


if __name__ == "__main__":
    main()
