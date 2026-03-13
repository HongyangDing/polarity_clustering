#!/bin/bash
#SBATCH -J shelly_cut
#SBATCH -p cpu
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH --get-user-env
#SBATCH --nodes=4
#SBATCH --ntasks=96
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=12:00:00

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

echo "[INFO] step1 cut_waveforms.py started at $(date)"
echo "[INFO] nodes=${SLURM_JOB_NUM_NODES:-1} ntasks=${SLURM_NTASKS:-1} cpus_per_task=${SLURM_CPUS_PER_TASK:-1}"
echo "[INFO] nodelist=${SLURM_JOB_NODELIST:-unknown}"
srun \
  --unbuffered \
  --ntasks="${SLURM_NTASKS}" \
  --cpu-bind=cores \
  --kill-on-bad-exit=1 \
  --label \
  python -u cut_waveforms.py
echo "[INFO] step1 cut_waveforms.py finished at $(date)"
