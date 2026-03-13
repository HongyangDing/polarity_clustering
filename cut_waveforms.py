import os
import socket

import numpy as np

import config
from lib.cut import iter_event_waveform_jobs, load_waveforms, read_fpha_list, read_fsta


def prepare_jobs():
    sta_dict = read_fsta(config.fsta)
    return list(iter_event_waveform_jobs(sta_dict.keys()))


def get_parallel_context():
    slurm_rank = os.environ.get("SLURM_PROCID")
    slurm_size = os.environ.get("SLURM_NTASKS")
    if slurm_rank is not None and slurm_size is not None:
        return int(slurm_rank), int(slurm_size), "slurm"

    try:
        from mpi4py import MPI
    except Exception:
        return 0, 1, "serial"

    comm = MPI.COMM_WORLD
    return comm.Get_rank(), comm.Get_size(), "mpi"


def cut_catalog_waveforms(dataset_name, catalog_path, output_dir, jobs_for_rank, rank):
    event_dic = read_fpha_list(catalog_path)
    event_ids = list(event_dic.keys())
    os.makedirs(output_dir, exist_ok=True)

    for idx, (sta_name, channel, phase_id) in enumerate(jobs_for_rank, start=1):
        print(f"[{dataset_name}] rank:{rank} {idx}/{len(jobs_for_rank)} {sta_name} {channel} {phase_id}")
        mat = load_waveforms(
            data_path=config.data_path,
            event_dic=event_dic,
            event_ids=event_ids,
            sta_name=sta_name,
            channel=channel,
            phase_id=phase_id,
        )
        mat_name = f"{sta_name}_{channel}_{phase_id}.npy"
        np.save(os.path.join(output_dir, mat_name), mat)


def main():
    rank, size, launch_mode = get_parallel_context()
    hostname = socket.gethostname()

    jobs = prepare_jobs()
    job_splits = np.array_split(np.array(jobs, dtype=object), size)
    jobs_for_rank = [tuple(job) for job in job_splits[rank].tolist()]
    print(
        f"[PARALLEL] mode={launch_mode} rank={rank} size={size} "
        f"host={hostname} jobs={len(jobs_for_rank)}"
    )

    datasets = [
        ("detected", config.fdetected, config.output_detected),
        ("temp", config.ftemp, config.output_temp),
    ]

    for dataset_name, catalog_path, output_dir in datasets:
        if not os.path.exists(catalog_path):
            raise FileNotFoundError(f"{dataset_name} catalog not found: {catalog_path}")
        cut_catalog_waveforms(
            dataset_name=dataset_name,
            catalog_path=catalog_path,
            output_dir=output_dir,
            jobs_for_rank=jobs_for_rank,
            rank=rank,
        )


if __name__ == "__main__":
    main()
