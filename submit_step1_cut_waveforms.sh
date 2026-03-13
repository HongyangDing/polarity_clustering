#!/bin/bash
#SBATCH -J shelly_cut
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=12:00:00

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

echo "[INFO] step1 cut_waveforms.py started at $(date)"
echo "[INFO] nodes=${SLURM_JOB_NUM_NODES:-1} ntasks=${SLURM_NTASKS:-1} cpus_per_task=${SLURM_CPUS_PER_TASK:-1}"
srun --cpu-bind=cores python -u cut_waveforms.py
echo "[INFO] step1 cut_waveforms.py finished at $(date)"
