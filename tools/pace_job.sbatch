#!/usr/bin/env bash
#SBATCH --account=gts-agarg35
#SBATCH --qos=embers
#SBATCH --partition=gpu-v100,gpu-rtx6000,gpu-a100
#SBATCH --exclude=atl1-1-01-002-2-0,atl1-1-01-002-4-0
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --job-name=libero-release
#SBATCH --output=logs/release-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
method="${2:?Use: sbatch tools/pace_job.sbatch MODE METHOD SEED [options]}"
if [ "$method" = a2a ]; then module load python/3.10.10; else module load python/3.8.19; fi
case "$method" in
 act) policy_env=libero-env;;
 dp) policy_env=dp-env;;
 baku) policy_env=baku-env;;
 a2a) policy_env=a2a-env;;
 *) exit 2;;
esac
export PYTHONPATH="$PWD:$PWD/third_party/LIBERO:$PWD/third_party/DiffusionPolicy:$PWD/third_party/A2A:$PWD/third_party/BAKU/baku"
export LIBERO_CONFIG_PATH="$PWD/configs/libero"
export TORCH_HOME="$HOME/scratch/robot_manipulation/torch-cache"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 NUMBA_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
unset DISPLAY
"$HOME/scratch/robot_manipulation/$policy_env/bin/python" -u tools/pace_entry.py "$@"
