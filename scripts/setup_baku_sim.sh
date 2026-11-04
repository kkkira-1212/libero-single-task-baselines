#!/usr/bin/env bash
set -euo pipefail
module load python/3.8.19
BAKU_PY="$HOME/scratch/robot_manipulation/baku-env/bin/python"
export PIP_CACHE_DIR="$HOME/scratch/robot_manipulation/pip-cache"
"$BAKU_PY" -m pip install -r configs/baku/requirements-simulation.txt
"$BAKU_PY" -m pip install --no-deps -e third_party/LIBERO
"$BAKU_PY" -m pip check
export LIBERO_CONFIG_PATH="$PWD/configs/libero"
export PYTHONPATH="$PWD/third_party/BAKU/baku:$PWD"
"$BAKU_PY" -c 'import mujoco, robosuite; from libero.libero.envs import OffScreenRenderEnv; from agent.baku import BCAgent; print("BAKU + LIBERO imports passed")'
"$BAKU_PY" -m pip freeze > "logs/baku-simulation-env-$SLURM_JOB_ID.freeze.txt"
