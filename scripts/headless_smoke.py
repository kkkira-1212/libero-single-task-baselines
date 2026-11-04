"""Single-task simulation check, no learned policy and no training."""
import argparse
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import socket
import subprocess
import time

import numpy as np
from PIL import Image
import torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
args = parser.parse_args()
out = Path(args.output)
out.mkdir(parents=True, exist_ok=True)
assert os.environ.get("SLURM_JOB_ID"), "Run through Slurm on a compute node."
assert os.environ.get("MUJOCO_GL") == "egl"
assert not os.environ.get("DISPLAY"), "Expected a headless process."
start = time.time()
report = {
    "job_id": os.environ["SLURM_JOB_ID"], "host": socket.gethostname(),
    "seed": 0, "policy": "scripted small Cartesian displacement; no learned model",
    "MUJOCO_GL": os.environ["MUJOCO_GL"],
    "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "versions": {p: metadata.version(p) for p in
                 ["libero", "torch", "numpy", "mujoco", "robosuite"]},
    "repo_commit": subprocess.check_output(
        ["git", "-C", "third_party/LIBERO", "rev-parse", "HEAD"], text=True).strip(),
    "passed": False,
}
env = None
try:
    report["cuda_available"] = torch.cuda.is_available()
    if report["cuda_available"]:
        report["gpu_name"] = torch.cuda.get_device_name(0)
        try:
            # Arithmetic probe only; no model inference or training.
            report["cuda_arithmetic_ok"] = (torch.ones(1, device="cuda") + 1).item() == 2
        except Exception as exc:
            report["cuda_probe_error"] = repr(exc)
    np.random.seed(0)
    suite = benchmark.get_benchmark_dict()["libero_object"]()
    task = suite.get_task(0)
    report["task"] = task.name
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    print("Creating task:", task.name, flush=True)
    env = OffScreenRenderEnv(bddl_file_name=str(bddl),
                             camera_heights=128, camera_widths=128,
                             render_gpu_device_id=0)
    env.seed(0)
    obs = env.reset()
    print("Reset passed", flush=True)
    obs = env.set_init_state(suite.get_task_init_states(0)[0])
    report["initial_state_id"] = 0
    report["observation_schema"] = {
        k: {"shape": list(np.asarray(v).shape), "dtype": str(np.asarray(v).dtype)}
        for k, v in obs.items()
    }
    low, high = env.env.action_spec
    assert low.shape == high.shape == (7,)
    report["action_low"] = low.tolist()
    report["action_high"] = high.tolist()
    report["image_statistics"] = []
    def save_images(observation, label):
        for key in ("agentview_image", "robot0_eye_in_hand_image"):
            frame = np.asarray(observation[key])
            assert frame.shape == (128, 128, 3) and frame.dtype == np.uint8
            assert float(frame.std()) > 1, "Blank or near-constant image"
            # MuJoCo's image origin is bottom-left; flip for saved PNGs.
            Image.fromarray(frame[::-1]).save(out / (label + "_" + key + ".png"))
            report["image_statistics"].append(
                {"label": label, "camera": key, "std": float(frame.std())})
    save_images(obs, "reset")
    initial_qpos = env.sim.data.qpos.copy()
    trajectory = []
    for step in range(30):
        action = np.zeros(7, dtype=np.float32)
        action[-1] = -1.0
        if step >= 10:
            action[0] = 0.05
        assert np.all(action >= low) and np.all(action <= high)
        obs, reward, done, info = env.step(action)
        assert np.isfinite(env.sim.data.qpos).all()
        trajectory.append({"step": step, "action": action.tolist(),
                           "reward": float(reward), "done": bool(done)})
        if step in (9, 19, 29):
            save_images(obs, "step_%03d" % (step + 1))
        if done:
            break
    report["steps"] = len(trajectory)
    report["qpos_delta_norm"] = float(np.linalg.norm(env.sim.data.qpos - initial_qpos))
    report["task_success_observed"] = bool(env.check_success())
    report["note"] = "Functional smoke test; not a policy success-rate evaluation."
    (out / "trajectory.json").write_text(json.dumps(trajectory, indent=2))
    report["passed"] = report["steps"] > 0
except Exception as exc:
    report["error"] = repr(exc)
    raise
finally:
    if env is not None:
        env.close()
    report["wall_seconds"] = time.time() - start
    (out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
