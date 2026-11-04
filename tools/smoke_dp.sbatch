#!/usr/bin/env bash
#SBATCH --account=gts-agarg35
#SBATCH --qos=embers
#SBATCH --partition=gpu-v100,gpu-rtx6000,gpu-a100
#SBATCH --exclude=atl1-1-01-002-2-0,atl1-1-01-002-4-0
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --job-name=delivery-smoke
#SBATCH --output=logs/delivery-smoke-%j.log
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
module load python/3.8.19
export PYTHONPATH="$PWD:$PWD/third_party/LIBERO:$PWD/third_party/DiffusionPolicy"
export LIBERO_CONFIG_PATH="$PWD/configs/libero"
export TORCH_HOME="$HOME/scratch/robot_manipulation/torch-cache"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 NUMBA_NUM_THREADS=4
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
unset DISPLAY
"$HOME/scratch/robot_manipulation/dp-env/bin/python" -u tools/smoke_dp.py
