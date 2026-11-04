#!/usr/bin/env bash
#SBATCH --account=gts-agarg35
#SBATCH --qos=embers
#SBATCH --mail-user=xinwang915@gatech.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=a2a-confirm
#SBATCH --partition=gpu-v100,gpu-rtx6000,gpu-a100
#SBATCH --exclude=atl1-1-01-002-2-0,atl1-1-01-002-4-0
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/a2a-eval-%j.log
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
"$HOME/scratch/robot_manipulation/a2a-env/bin/python" -c 'import torch; n=torch.cuda.get_device_name(0); print("Actual GPU:",n,flush=True); assert "RTX6000" in n.upper().replace(" ",""),"Requested RTX6000 hardware not satisfied"'
"$HOME/scratch/robot_manipulation/a2a-env/bin/python" -u experiments/a2a_confirm_eval.py "$@"
