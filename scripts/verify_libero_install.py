"""Verify installation and task files only; no simulation or model execution."""
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import sys

project = Path(__file__).resolve().parents[1]
os.environ["LIBERO_CONFIG_PATH"] = str(project / "configs/libero")
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

suite = benchmark.get_benchmark_dict()["libero_object"]()
task = suite.get_task(0)
bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
initial = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
assert bddl.is_file(), bddl
assert initial.is_file(), initial
assert Path(get_libero_path("assets")).is_dir()
print(json.dumps({
    "python": sys.version,
    "packages": {name: metadata.version(name) for name in
                 ["libero", "torch", "numpy", "mujoco", "robosuite", "bddl"]},
    "suite": "libero_object",
    "task": task.name,
    "bddl_file": str(bddl),
    "initial_states_file": str(initial),
    "environment_class_imported": OffScreenRenderEnv.__name__,
    "simulation_executed": False,
    "gpu_tested": False
}, indent=2))
