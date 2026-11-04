#!/usr/bin/env bash
#SBATCH --account=gts-agarg35
#SBATCH --qos=embers
#SBATCH --partition=gpu-v100,gpu-rtx6000,gpu-a100
#SBATCH --exclude=atl1-1-01-002-2-0,atl1-1-01-002-4-0
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --mail-user=xinwang915@gatech.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=baseline-causal
#SBATCH --output=logs/baseline-causal-%j.log
set -eo pipefail
cd /storage/home/hcoda1/5/xwang3405/research/robot_manipulation
source scripts/activate-causal-baseline.sh "$1"
export CAUSAL_GATE_JOB="$4"
extra=()
if [ -n "${5:-}" ]; then extra=(--eval-protocol "$5"); fi
case "$1" in
  act) python -u experiments/act_eval.py --checkpoint "$HOME/scratch/robot_manipulation/act-pilot/$3/latest.pt" --minimum-training-step 10000 ;;
  dp) python -u experiments/dp_run.py eval --config "$2" --training-job "$3" "${extra[@]}" ;;
  baku) python -u experiments/baku_run.py eval --config "$2" --training-job "$3" "${extra[@]}" ;;
esac
