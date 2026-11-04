#!/usr/bin/env bash
#SBATCH --account=gts-agarg35
#SBATCH --qos=embers
#SBATCH --partition=gpu-v100,gpu-rtx6000,gpu-a100
#SBATCH --exclude=atl1-1-01-002-2-0,atl1-1-01-002-4-0
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --mail-user=xinwang915@gatech.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=dp-seed1-exec1
#SBATCH --output=logs/dp-eval-%j.log
set -eo pipefail
cd /storage/home/hcoda1/5/xwang3405/research/robot_manipulation
source scripts/activate-dp.sh
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=4
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
unset DISPLAY
hostname
nvidia-smi
python -u experiments/dp_run.py eval --config configs/diffusion_policy/train_seed1.json --checkpoint "/storage/home/hcoda1/5/xwang3405/scratch/robot_manipulation/dp-training/14353584/latest.pt" --eval-protocol configs/diffusion_policy/eval_exec1_seed1.json --resume-eval-report artifacts/dp-eval/14463330/result.json
