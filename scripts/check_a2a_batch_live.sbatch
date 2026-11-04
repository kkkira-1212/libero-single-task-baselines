#!/usr/bin/env bash
#SBATCH --account=gts-agarg35
#SBATCH --qos=embers
#SBATCH --mail-user=xinwang915@gatech.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=a2a-integration
#SBATCH --partition=gpu-v100,gpu-rtx6000,gpu-a100,gpu-l40s
#SBATCH --exclude=atl1-1-01-002-2-0,atl1-1-01-002-4-0
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=logs/a2a-integration-%j.log
set -eo pipefail
cd /storage/home/hcoda1/5/xwang3405/research/robot_manipulation
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4
module load python/3.10.10
export PYTHONPATH="$PWD:$PWD/third_party/A2A:$PWD/third_party/LIBERO"
export LIBERO_CONFIG_PATH="$PWD/configs/libero"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
export NUMBA_NUM_THREADS=4
unset DISPLAY
"$HOME/scratch/robot_manipulation/a2a-env/bin/python" -u scripts/check_a2a_batch_live.py
