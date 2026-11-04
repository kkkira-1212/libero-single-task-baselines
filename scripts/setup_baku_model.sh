#!/usr/bin/env bash
set -euo pipefail
module load python/3.8.19
BAKU_PY="$HOME/scratch/robot_manipulation/baku-env/bin/python"
export PIP_CACHE_DIR="$HOME/scratch/robot_manipulation/pip-cache"
"$BAKU_PY" -m pip install robomimic==0.2.0 diffusers==0.11.1
"$BAKU_PY" -m pip check
export PYTHONPATH="$PWD/third_party/BAKU/baku"
"$BAKU_PY" -c 'from agent.baku import BCAgent; print("Official BAKU model imports passed")'
"$BAKU_PY" -m pip freeze > "logs/baku-model-env-$SLURM_JOB_ID.freeze.txt"
