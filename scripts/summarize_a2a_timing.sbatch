#!/usr/bin/env bash
#SBATCH --account=gts-agarg35
#SBATCH --qos=embers
#SBATCH --mail-user=xinwang915@gatech.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=a2a-timing-results
#SBATCH --partition=cpu-small
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/a2a-timing-results-%j.log
set -eo pipefail
cd /storage/home/hcoda1/5/xwang3405/research/robot_manipulation
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4
python3 -u scripts/summarize_a2a_timing.py
