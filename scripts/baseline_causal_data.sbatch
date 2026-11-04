#!/usr/bin/env bash
#SBATCH --account=gts-agarg35
#SBATCH --qos=embers
#SBATCH --partition=cpu-small
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --mail-user=xinwang915@gatech.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=baseline-causal
#SBATCH --output=logs/baseline-causal-%j.log
set -eo pipefail
cd /storage/home/hcoda1/5/xwang3405/research/robot_manipulation
source scripts/activate-causal-baseline.sh "$1"
python -u scripts/check_baseline_causal.py --method "$1" --data-only
