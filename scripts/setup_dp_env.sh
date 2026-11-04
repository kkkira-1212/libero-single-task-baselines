#!/usr/bin/env bash
set -euo pipefail
module load python/3.8.19
DP_ENV="$HOME/scratch/robot_manipulation/dp-env"
export PIP_CACHE_DIR="$HOME/scratch/robot_manipulation/pip-cache"
if test ! -f "$DP_ENV/pyvenv.cfg"; then
    python -m venv "$DP_ENV"
fi
"$DP_ENV/bin/python" -m pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
"$DP_ENV/bin/python" -m pip install -r third_party/LIBERO/requirements.txt -r configs/diffusion_policy/requirements-policy.txt
"$DP_ENV/bin/python" -m pip install --no-deps -e third_party/LIBERO
"$DP_ENV/bin/python" -m pip check
"$DP_ENV/bin/python" -m pip freeze > logs/dp-environment.freeze.txt
export PYTHONPATH="$PWD:$PWD/third_party/DiffusionPolicy"
"$DP_ENV/bin/python" -u scripts/verify_dp_imports.py
