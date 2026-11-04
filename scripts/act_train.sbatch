#!/usr/bin/env bash
#SBATCH --job-name=act-train-seed0
#SBATCH --account=gts-agarg35
#SBATCH --qos=embers
#SBATCH --partition=gpu-v100,gpu-rtx6000,gpu-a100,gpu-l40s
#SBATCH --exclude=atl1-1-01-002-2-0,atl1-1-01-002-4-0
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --output=logs/act-train-%j.log
#SBATCH --mail-user=xinwang915@gatech.edu
#SBATCH --mail-type=BEGIN,END,FAIL
set -eo pipefail
cd /storage/home/hcoda1/5/xwang3405/research/robot_manipulation
source scripts/activate-libero.sh
export TORCH_HOME="$HOME/scratch/robot_manipulation/torch-cache"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
date -Is
hostname
nvidia-smi
python -u experiments/act_pilot.py --config configs/act/train_seed0.json "$@" &
train_pid=$!
trap 'kill -USR1 "$train_pid" 2>/dev/null || true' USR1 TERM
wait "$train_pid" || { result=$?; wait "$train_pid" 2>/dev/null || exit "$result"; }
