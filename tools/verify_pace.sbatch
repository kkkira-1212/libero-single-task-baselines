#!/usr/bin/env bash
#SBATCH --account=gts-agarg35
#SBATCH --qos=embers
#SBATCH --partition=cpu-small
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --job-name=delivery-check
#SBATCH --output=logs/delivery-check-%j.log
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
module load python/3.8.19
python3 tools/bootstrap_upstream.py --check-only
python3 tools/verify_delivery.py
python3 tools/verify_pace_assets.py
export PYTHONPATH="$PWD:$PWD/third_party/LIBERO:$PWD/third_party/DiffusionPolicy:$PWD/third_party/A2A:$PWD/third_party/BAKU/baku"
export LIBERO_CONFIG_PATH="$PWD/configs/libero"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 NUMBA_NUM_THREADS=2
for method in act dp baku; do
 case "$method" in act) policy_env=libero-env;; dp) policy_env=dp-env;; baku) policy_env=baku-env;; esac
 "$HOME/scratch/robot_manipulation/$policy_env/bin/python" -u tools/pace_entry.py data-check "$method" 0
 mkdir -p "artifacts/release-check/$SLURM_JOB_ID"
 cp "artifacts/baseline-causal-check/$SLURM_JOB_ID/result.json" "artifacts/release-check/$SLURM_JOB_ID/data-$method.json"
done
module load python/3.10.10
"$HOME/scratch/robot_manipulation/a2a-env/bin/python" -u tools/pace_entry.py data-check a2a 0
python3 - <<'PY2'
import json,os
from pathlib import Path
root=Path.cwd();job=os.environ['SLURM_JOB_ID']
paths=[root/'artifacts/release-assets'/job/'result.json',root/'artifacts/a2a-causal-data'/job/'result.json',root/'artifacts/release-check'/job/'data-act.json',root/'artifacts/release-check'/job/'data-dp.json',root/'artifacts/release-check'/job/'data-baku.json']
assert all(p.exists() and json.loads(p.read_text())['passed'] for p in paths)
out=root/'artifacts/release-check'/job;out.mkdir(parents=True,exist_ok=True)
(out/'result.json').write_text(json.dumps(dict(passed=True,job=job,scope='Post-maintenance packaged data interfaces and checkpoint/dataset integrity; no training or scored rollout',checks=[str(p.relative_to(root)) for p in paths]),indent=2)+'\n')
print('DELIVERY_CHECK_PASSED')
PY2
