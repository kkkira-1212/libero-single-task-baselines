#!/usr/bin/env bash
# Run from the project root after creating the independent environment.
# Downloads/installs packages only; does not submit jobs or run simulation.
set -euo pipefail
source scripts/activate-libero.sh
python -m pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
python -m pip install -r third_party/LIBERO/requirements.txt -r configs/libero-extra-constraints.txt
python -m pip install -e third_party/LIBERO
python -m pip check
