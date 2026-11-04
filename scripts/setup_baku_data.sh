#!/usr/bin/env bash
set -euo pipefail
module load python/3.8.19
BAKU_ENV="$HOME/scratch/robot_manipulation/baku-env"
export PIP_CACHE_DIR="$HOME/scratch/robot_manipulation/pip-cache"
export HF_HOME="$HOME/scratch/robot_manipulation/hf-cache"
if test ! -f "$BAKU_ENV/pyvenv.cfg"; then python -m venv "$BAKU_ENV"; fi
"$BAKU_ENV/bin/python" -m pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
"$BAKU_ENV/bin/python" -m pip install -r configs/baku/requirements-data.txt
"$BAKU_ENV/bin/python" -m pip check
"$BAKU_ENV/bin/python" -m pip freeze > "logs/baku-data-env-$SLURM_JOB_ID.freeze.txt"
"$BAKU_ENV/bin/python" -u scripts/prepare_baku_data.py
