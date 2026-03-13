#!/bin/bash
#SBATCH -J shelly_cc
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

set -euo pipefail

WORKPATH="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$WORKPATH"

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate obspy
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CC_TEMPLATE_BATCH_SIZE="${CC_TEMPLATE_BATCH_SIZE:-4}"
export CC_RELATIVE_POLARITY_MODE="${CC_RELATIVE_POLARITY_MODE:-shelly2016_proxy}"
export CC_SVD_MODE="${CC_SVD_MODE:-reduced}"
export CC_SVD_NITER="${CC_SVD_NITER:-2}"

echo "[INFO] step2 cc_cal.py started at $(date)"
echo "[INFO] nodes=${SLURM_JOB_NUM_NODES:-1} ntasks=${SLURM_NTASKS:-1} cpus_per_task=${SLURM_CPUS_PER_TASK:-1}"
echo "[INFO] gpus_on_node=${SLURM_GPUS_ON_NODE:-unknown} job_gpus=${SLURM_JOB_GPUS:-unknown}"
echo "[INFO] CC_RELATIVE_POLARITY_MODE=${CC_RELATIVE_POLARITY_MODE}"
echo "[INFO] CC_TEMPLATE_BATCH_SIZE=${CC_TEMPLATE_BATCH_SIZE} CC_SVD_MODE=${CC_SVD_MODE}"
echo "[INFO] If you have multiple GPUs available, increase --nodes and keep 1 MPI rank per GPU."
srun --cpu-bind=cores python -u cc_cal.py
echo "[INFO] step2 cc_cal.py finished at $(date)"
