#!/usr/bin/env bash
#SBATCH --account=gts-agarg35
#SBATCH --qos=embers
#SBATCH --partition=cpu-small
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --mail-user=xinwang915@gatech.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=baku-data-check
#SBATCH --output=logs/baku-data-check-%j.log
set -eo pipefail
cd /storage/home/hcoda1/5/xwang3405/research/robot_manipulation
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=2
module load python/3.8.19
"$HOME/scratch/robot_manipulation/baku-env/bin/python" -u scripts/check_baku_data.py --prepare-job "$1"
