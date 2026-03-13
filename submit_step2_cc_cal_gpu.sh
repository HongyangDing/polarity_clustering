#!/bin/bash
#SBATCH -J shelly_cc
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH --partition=gpu
#SBATCH --get-user-env
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00

set -euo pipefail

WORKPATH="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$WORKPATH"

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    . "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    . "$HOME/miniforge3/etc/profile.d/conda.sh"
fi

if command -v conda >/dev/null 2>&1; then
    conda activate obspy
fi

# Program behavior defaults belong in config.py; this script only sets runtime environment.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "[INFO] step2 cc_cal.py started at $(date)"
echo "[INFO] nodes=${SLURM_JOB_NUM_NODES:-1} ntasks=${SLURM_NTASKS:-1} cpus_per_task=${SLURM_CPUS_PER_TASK:-1}"
echo "[INFO] gpus_on_node=${SLURM_GPUS_ON_NODE:-unknown} job_gpus=${SLURM_JOB_GPUS:-unknown}"
echo "[INFO] Step 2 algorithm parameters come from config.py"
nvidia-smi -L || true
srun --cpu-bind=cores python -u cc_cal.py
echo "[INFO] step2 cc_cal.py finished at $(date)"
