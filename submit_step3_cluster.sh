#!/bin/bash
#SBATCH -J shelly_cluster
#SBATCH -p cpu
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH --get-user-env
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

WORKPATH="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$WORKPATH"
mkdir -p "$WORKPATH/.cache"

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
export XDG_CACHE_HOME="$WORKPATH/.cache"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "[INFO] step3 cluster.py started at $(date)"
srun --cpu-bind=cores python -u cluster.py
echo "[INFO] step3 cluster.py finished at $(date)"
