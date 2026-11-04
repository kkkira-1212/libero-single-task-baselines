#!/usr/bin/env bash
#SBATCH --job-name=libero-smoke
#SBATCH --account=gts-agarg35
#SBATCH --qos=embers
#SBATCH --partition=gpu-v100,gpu-rtx6000,gpu-a100,gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=logs/headless-%j.log
set -eo pipefail
cd /storage/home/hcoda1/5/xwang3405/research/robot_manipulation
source scripts/activate-libero.sh
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=4
unset NUMBA_DISABLE_JIT
unset DISPLAY
date -Is
hostname
nvidia-smi
scontrol show job "$SLURM_JOB_ID"
python -u scripts/headless_smoke.py --output "artifacts/headless/$SLURM_JOB_ID"
