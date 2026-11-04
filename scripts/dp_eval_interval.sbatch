#!/usr/bin/env bash
#SBATCH --account=gts-agarg35
#SBATCH --qos=embers
#SBATCH --partition=gpu-rtx6000
#SBATCH --exclude=atl1-1-01-002-2-0,atl1-1-01-002-4-0
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --mail-user=xinwang915@gatech.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=dp-interval-eval
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
python -u experiments/dp_run.py eval --checkpoint "$HOME/scratch/robot_manipulation/frozen/dp_seed0_step3000/checkpoint.pt" --eval-protocol "$1"
