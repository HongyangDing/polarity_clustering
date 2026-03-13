#!/bin/bash
#SBATCH -J shelly_sel
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
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
    conda activate obspy
fi

export PYTHONUNBUFFERED=1

echo "[INFO] step0 sel_temp.py started at $(date)"
python -u sel_temp.py
echo "[INFO] step0 sel_temp.py finished at $(date)"
