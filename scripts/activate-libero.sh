# Source from a PACE shell: source scripts/activate-libero.sh
module load python/3.8.19
source "$HOME/scratch/robot_manipulation/libero-env/bin/activate"
export LIBERO_CONFIG_PATH=/storage/home/hcoda1/5/xwang3405/research/robot_manipulation/configs/libero
export PIP_CACHE_DIR="$HOME/scratch/robot_manipulation/pip-cache"
