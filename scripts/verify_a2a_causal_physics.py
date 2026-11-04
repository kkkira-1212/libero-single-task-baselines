"""Zero-update causal-index and actual OSC goal/motion audit, CPU without rendering."""
import hashlib, json, os, sys, time, traceback
from pathlib import Path
import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from adapters.libero_causal_contract import CONTRACT_VERSION, causal_indices

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/a2a-causal-physics" / os.environ["SLURM_JOB_ID"]
OUT.mkdir(parents=True, exist_ok=True)
report = {"completed": False, "training_updates": 0, "contract": CONTRACT_VERSION}
start = time.time()
env = None

def write():
    (OUT / "result.json").write_text(json.dumps(report, indent=2) + "\n")

def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))

try:
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs.env_wrapper import ControlEnv
    path = Path.home() / "scratch/robot_manipulation/datasets/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5"
    report["dataset_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    checks = {"episodes": 0, "causal_windows": 0, "stored_postobs_vs_next_prestate_max_abs": 0.0}
    with h5py.File(path, "r") as f:
        for name in f["data"]:
            d = f["data"][name]
            q = np.concatenate([d["obs/joint_states"][:], d["obs/gripper_states"][:]], axis=1)
            states = d["states"][:]
            error = float(np.max(np.abs(q[:-1] - states[1:, 1:10])))
            checks["stored_postobs_vs_next_prestate_max_abs"] = max(checks["stored_postobs_vs_next_prestate_max_abs"], error)
            assert error < 1e-7, (name, error)
            n = len(q)
            for t in sorted(set([0, 1, 7, n // 2, n - 2])):
                idx = causal_indices(n, t)
                assert idx["history_indices"][-1] == t
                assert idx["future_indices"][0] == t + 1
                assert np.all(idx["history_indices"] <= t)
                assert np.all(idx["future_indices"] > t)
                assert idx["future_valid"][0]
                assert np.all(idx["future_indices"] < n)
                checks["causal_windows"] += 1
            try:
                causal_indices(n, n - 1)
            except ValueError:
                pass
            else:
                raise AssertionError("Last observation must not invent a next command")
            checks["episodes"] += 1
        report["static_checks"] = checks
        suite = benchmark.get_benchmark_dict()["libero_object"]()
        task = suite.get_task(0)
        env = ControlEnv(bddl_file_name=str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file),
                         use_camera_obs=False, has_renderer=False, has_offscreen_renderer=False)
        report["replays"] = []
        for name in ("demo_28", "demo_36"):
            d = f["data"][name]
            actions = d["actions"][:]
            env.seed(0)
            env.reset()
            env.set_init_state(d["states"][0])
            controller = env.env.robots[0].controller
            assert controller.use_delta and controller.impedance_mode == "fixed"
            report["controller"] = {
                "class": type(controller).__name__, "use_delta": bool(controller.use_delta),
                "input_min": np.asarray(controller.input_min).tolist(),
                "input_max": np.asarray(controller.input_max).tolist(),
                "output_min": np.asarray(controller.output_min).tolist(),
                "output_max": np.asarray(controller.output_max).tolist(),
                "position_limits": None if controller.position_limits is None else np.asarray(controller.position_limits).tolist(),
                "orientation_limits": None if controller.orientation_limits is None else np.asarray(controller.orientation_limits).tolist(),
                "rotation_rule": "R_goal = Exp(scaled_axis_angle_world) @ R_current; exactly-zero rotation retains previous goal"
            }
            captures = []
            original = controller.set_goal
            def capture(action, *args, **kwargs):
                previous = None if controller.goal_ori is None else controller.goal_ori.copy()
                result = original(action, *args, **kwargs)
                captures.append({
                    "action": np.asarray(action).copy(), "pre_pos": controller.ee_pos.copy(),
                    "pre_rot": controller.ee_ori_mat.copy(), "goal_pos": controller.goal_pos.copy(),
                    "goal_rot": controller.goal_ori.copy(), "previous_goal_rot": previous,
                })
                return result
            controller.set_goal = capture
            rows = []
            try:
                for t, action in enumerate(actions):
                    captures.clear()
                    _, _, done, _ = env.step(action)
                    assert len(captures) == 1, len(captures)
                    c = captures[0]
                    controller.update(force=True)
                    post_pos, post_rot = controller.ee_pos.copy(), controller.ee_ori_mat.copy()
                    scaled = controller.scale_action(action[:6])
                    assert np.max(np.abs(c["action"] - action[:6])) < 1e-7
                    scale = np.asarray(controller.action_scale)
                    offset = np.asarray(controller.action_output_transform)
                    inp = np.asarray(controller.action_input_transform)
                    measured = np.r_[post_pos - c["pre_pos"], Rotation.from_matrix(post_rot @ c["pre_rot"].T).as_rotvec()]
                    reconstructed = np.r_[c["goal_pos"] - c["pre_pos"], Rotation.from_matrix(c["goal_rot"] @ c["pre_rot"].T).as_rotvec()]
                    nonzero_rotation = bool(np.any(action[3:6] != 0))
                    expected_rot = (Rotation.from_rotvec(scaled[3:]).as_matrix() @ c["pre_rot"]
                                    if nonzero_rotation else c["previous_goal_rot"])
                    assert expected_rot is not None
                    goal_rot_error = float(np.max(np.abs(c["goal_rot"] - expected_rot)))
                    goal_pos_error = float(np.max(np.abs(c["goal_pos"] - (c["pre_pos"] + scaled[:3]))))
                    inverse = (reconstructed - offset) / scale + inp
                    inverse_error = float(np.max(np.abs(inverse[:6 if nonzero_rotation else 3] - action[:6 if nonzero_rotation else 3])))
                    rows.append({
                        "t": t, "action": action.tolist(), "scaled_command": scaled.tolist(),
                        "measured_delta": measured.tolist(), "measured_normalized_equivalent": ((measured-offset)/scale+inp).tolist(),
                        "goal_inverse": inverse.tolist(), "rotation_command_nonzero": nonzero_rotation,
                        "goal_pos_error": goal_pos_error, "goal_rot_error": goal_rot_error,
                        "inverse_error": inverse_error, "pre_pos": c["pre_pos"].tolist(),
                        "post_pos": post_pos.tolist(), "goal_pos": c["goal_pos"].tolist(),
                    })
                    success = bool(env.check_success())
                    if success or done:
                        break
            finally:
                controller.set_goal = original
            commands = np.array([r["action"][:6] for r in rows])
            motion = np.array([r["measured_normalized_equivalent"] for r in rows])
            pos_error = max(r["goal_pos_error"] for r in rows)
            rot_error = max(r["goal_rot_error"] for r in rows)
            inv_error = max(r["inverse_error"] for r in rows)
            assert pos_error < 1e-7 and rot_error < 1e-6 and inv_error < 1e-5, (pos_error, rot_error, inv_error)
            item = {
                "demo": name, "success": success, "steps": len(rows),
                "controller_goal_position_max_abs": pos_error,
                "controller_goal_rotation_matrix_max_abs": rot_error,
                "goal_inverse_normalized_command_max_abs": inv_error,
                "measured_vs_command_translation_rmse_normalized": rmse(motion[:, :3], commands[:, :3]),
                "measured_vs_command_rotation_rmse_normalized": rmse(motion[:, 3:], commands[:, 3:]),
                "measured_vs_command_translation_rmse_meters": rmse(np.array([r["measured_delta"][:3] for r in rows]), np.array([r["scaled_command"][:3] for r in rows])),
                "zero_rotation_goal_hold_steps": sum(not r["rotation_command_nonzero"] for r in rows),
                "gripper_note": "Measured gripper qpos is not binary command; no inverse grip mapping claimed."
            }
            report["replays"].append(item)
            (OUT / (name + "_trajectory.json")).write_text(json.dumps(rows) + "\n")
            write()
            print(json.dumps(item), flush=True)
    report["completed"] = True
    report["interpretation"] = "Causal timing and OSC goal inverse verified, not evidence a new source representation improves A2A. Measured motion is executed response, not commanded goal. No training adapter was switched."
except Exception as exc:
    report["error"] = repr(exc)
    traceback.print_exc()
finally:
    if env is not None:
        env.close()
    report["wall_seconds"] = time.time() - start
    write()
    print(json.dumps(report, indent=2), flush=True)
sys.exit(0 if report["completed"] else 1)
