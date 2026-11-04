#!/usr/bin/env bash
set -euo pipefail
module load python/3.10.10
A2A_ENV="$HOME/scratch/robot_manipulation/a2a-env"
export PIP_CACHE_DIR="$HOME/scratch/robot_manipulation/pip-cache"
if test ! -f "$A2A_ENV/pyvenv.cfg";then python -m venv "$A2A_ENV";fi
"$A2A_ENV/bin/python" -m pip install --upgrade "pip<26"
"$A2A_ENV/bin/python" -m pip install torch==2.2.2+cu118 torchvision==0.17.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
"$A2A_ENV/bin/python" -m pip install -c configs/a2a/requirements-runtime.txt -r configs/a2a/requirements-runtime.txt
"$A2A_ENV/bin/python" -m pip install --no-deps -e third_party/LIBERO
"$A2A_ENV/bin/python" -m pip check
export PYTHONPATH="$PWD:$PWD/third_party/A2A:$PWD/third_party/LIBERO"
export LIBERO_CONFIG_PATH="$PWD/configs/libero"
"$A2A_ENV/bin/python" -u scripts/check_a2a_imports.py
"$A2A_ENV/bin/python" -m pip freeze > "logs/a2a-env-$SLURM_JOB_ID.freeze.txt"
