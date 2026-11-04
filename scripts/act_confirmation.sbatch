#!/usr/bin/env bash
#SBATCH --job-name=act-confirm-frozen
#SBATCH --account=gts-agarg35
#SBATCH --qos=embers
#SBATCH --partition=gpu-v100,gpu-rtx6000,gpu-a100,gpu-l40s
#SBATCH --exclude=atl1-1-01-002-2-0,atl1-1-01-002-4-0
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --signal=B:USR1@120
#SBATCH --output=logs/act-confirm-%j.log
#SBATCH --mail-user=xinwang915@gatech.edu
#SBATCH --mail-type=BEGIN,END,FAIL
set -eo pipefail
cd /storage/home/hcoda1/5/xwang3405/research/robot_manipulation
source scripts/activate-libero.sh
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4
export NUMBA_NUM_THREADS=4
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
unset DISPLAY
date -Is
hostname
nvidia-smi
python -u experiments/act_eval.py --checkpoint "$HOME/scratch/robot_manipulation/frozen/act_seed0_step10000/checkpoint.pt" --minimum-training-step 10000 --protocol configs/act/confirmation_frozen.json &
eval_pid=$!
trap 'kill -USR1 "$eval_pid" 2>/dev/null || true' USR1 TERM
wait "$eval_pid" || { result=$?; wait "$eval_pid" 2>/dev/null || exit "$result"; }
