#!/usr/bin/env bash
#SBATCH --job-name=cuda-check
#SBATCH --account=gts-agarg35
#SBATCH --qos=embers
#SBATCH --partition=gpu-v100,gpu-rtx6000,gpu-a100,gpu-l40s
#SBATCH --exclude=atl1-1-01-002-2-0,atl1-1-01-002-4-0
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/cuda-%j.log
#SBATCH --mail-user=xinwang915@gatech.edu
#SBATCH --mail-type=BEGIN,END,FAIL
set -eo pipefail
cd /storage/home/hcoda1/5/xwang3405/research/robot_manipulation
source scripts/activate-libero.sh
export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
date -Is
hostname
nvidia-smi
scontrol show job "$SLURM_JOB_ID"
python -u scripts/check_cuda.py
