#!/usr/bin/env bash
#SBATCH --account=gts-agarg35
#SBATCH --qos=embers
#SBATCH --mail-user=xinwang915@gatech.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=a2a-import-retry
#SBATCH --partition=cpu-small
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/a2a-env-%j.log
set -eo pipefail
cd /storage/home/hcoda1/5/xwang3405/research/robot_manipulation
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4
module load python/3.10.10
export PYTHONPATH="$PWD:$PWD/third_party/A2A:$PWD/third_party/LIBERO"
export LIBERO_CONFIG_PATH="$PWD/configs/libero"
export PIP_CACHE_DIR="$HOME/scratch/robot_manipulation/pip-cache"
"$HOME/scratch/robot_manipulation/a2a-env/bin/python" -m pip install termcolor==2.4.0
"$HOME/scratch/robot_manipulation/a2a-env/bin/python" -m pip check
"$HOME/scratch/robot_manipulation/a2a-env/bin/python" -u scripts/check_a2a_imports.py
"$HOME/scratch/robot_manipulation/a2a-env/bin/python" -m pip freeze > "logs/a2a-env-$SLURM_JOB_ID.freeze.txt"
