# Source on a compute node: source scripts/activate-causal-baseline.sh METHOD
case "$1" in
  act) source scripts/activate-libero.sh ;;
  dp) source scripts/activate-dp.sh ;;
  baku)
    module load python/3.8.19
    source "$HOME/scratch/robot_manipulation/baku-env/bin/activate"
    ;;
  *) echo "Unknown baseline method" >&2; return 2 ;;
esac
export PYTHONPATH="$PWD:$PWD/third_party/DiffusionPolicy:$PWD/third_party/LIBERO${PYTHONPATH:+:$PYTHONPATH}"
export LIBERO_CONFIG_PATH="$PWD/configs/libero"
export TORCH_HOME="$HOME/scratch/robot_manipulation/torch-cache"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4
export NUMBA_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=0
unset DISPLAY
