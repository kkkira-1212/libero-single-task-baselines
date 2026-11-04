# Source from project root on PACE.
module load python/3.8.19
source "$HOME/scratch/robot_manipulation/dp-env/bin/activate"
export PYTHONPATH="/storage/home/hcoda1/5/xwang3405/research/robot_manipulation:/storage/home/hcoda1/5/xwang3405/research/robot_manipulation/third_party/DiffusionPolicy"
export LIBERO_CONFIG_PATH=/storage/home/hcoda1/5/xwang3405/research/robot_manipulation/configs/libero
export TORCH_HOME="$HOME/scratch/robot_manipulation/torch-cache"
