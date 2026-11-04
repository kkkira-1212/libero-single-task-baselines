#!/usr/bin/env bash
#SBATCH --account=gts-agarg35
#SBATCH --qos=embers
#SBATCH --mail-user=xinwang915@gatech.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=a2a-seeds-cpu-fast
#SBATCH --partition=cpu-small
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/a2a-seeds-cpu-fast-%j.log
set -eo pipefail
cd /storage/home/hcoda1/5/xwang3405/research/robot_manipulation
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4
module load python/3.10.10
export PYTHONPATH="$PWD:$PWD/third_party/A2A:$PWD/third_party/LIBERO"
export LIBERO_CONFIG_PATH="$PWD/configs/libero"
export A2A_COMPARE_CPU_FAST=1
"$HOME/scratch/robot_manipulation/a2a-env/bin/python" -u scripts/compare_a2a_seeds.py
