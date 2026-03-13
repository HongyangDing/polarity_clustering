#!/bin/bash
#SBATCH -J shelly_sel
#SBATCH -p cpu
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH --get-user-env
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00

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

echo "[INFO] step0 sel_temp.py started at $(date)"
srun --cpu-bind=cores python -u sel_temp.py
echo "[INFO] step0 sel_temp.py finished at $(date)"
