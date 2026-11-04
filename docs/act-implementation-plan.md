# ACT on LIBERO — source inspection and adaptation plan

## Status
Data adapter implemented and real-data checks passed. CPU model structure check passed (9D state / 7D action / 20-step chunk); no ACT forward/backward, training or online evaluation yet.
Author ACT source checked out at third_party/ACT, commit 742c753c0d4a5d87076c8f69e5628c79a8cc5488.
Source: https://github.com/tonyzhaozh/act
Author project: https://tonyzhaozh.github.io/aloha/
The mentor-linked aloha repository supplies the robot platform; the author project links the separate ACT learning/simulation repository.

## Implementation decision
Start from the author ACT model, retain CVAE, chunk decoder and masked L1 + KL objective, and add explicit LIBERO data/evaluation adapters.
This is ACT adapted to LIBERO, not reproduction of the author's bimanual paper task scores.
Before coding, check whether an established LIBERO implementation offers a smaller verified adaptation. LeRobot is a maintained candidate, but its dependencies and LIBERO protocol have not been audited here; do not mix its claims/configs with the author code.

## Confirmed incompatibilities in author source
- detr/models/detr_vae.py: build() fixes state_dim=14; action head uses that dimension. Robot-state, encoder-action and encoder-state projections also hardcode 14. LIBERO requires distinct configurable observation and action dimensions.
- utils.py expects one episode_N.hdf5 per episode, observations/qpos and action fields; downloaded LIBERO HDF5 groups demonstrations in one file with different field names.
- utils.py applies a real-robot one-step offset in one branch; do not copy that offset to LIBERO without evidence.
- get_norm_stats stacks entire episodes (assumes equal lengths) and uses all episodes. Our trajectories vary in length; compute streaming/concatenated stats using training demonstrations only.
- detr/main.py builder parses global CLI arguments and calls model.cuda(). Adapter should separate configuration, model construction and explicit device selection.
- Dependencies include optional IPython/debug imports; original demo simulation requires dm_control, which is not required to execute LIBERO. Audit imports before installing packages.
- Backbone initialization may download weights; inspect and make this explicit before runtime.

## Initial data contract (data checks passed; actual online parity pending)
- Task: existing LIBERO-Object alphabet-soup-to-basket.
- Cameras: agentview_rgb and eye_in_hand_rgb, uint8 128x128 RGB -> float NCHW; preserve documented preprocessing and verify online image orientation.
- State: concatenate joint_states (7) and gripper_states (2), total 9 dimensions. Online mapping robot0_joint_pos + robot0_gripper_qpos.
- Target action: existing 7D delta end-effector pose/gripper command. This differs from original ACT joint-position output; keep controller semantics explicit.
- Model state_dim=9, action_dim=7 independently, no zero-padding to pretend a 14D bimanual robot.
- Whole-episode split written: 40 train / 10 validation, seed 0, configs/act/split.json. Training-only statistics in configs/act/normalization.json; reuse split across baselines.
- Normalize state/actions with training-split statistics only; checkpoint stores statistics and preprocessing.
- Sample current observation at t and future actions beginning at t; mask padding beyond episode end. Never cross episode boundaries.
- Chunk length and execution/replanning schedule remain to be selected from model/protocol inspection; do not copy 100 steps at 50 Hz blindly to a 20 Hz task.
- Record whether temporal ensembling is used; preserve correct time alignment and explicit validity masks.

## Validation gates
1. Freeze schema, split, normalization and image/action mapping.
2. Check real model forward/backward on one GPU batch; check finite losses/gradients and checkpoint save/reload.
3. Fit a small training subset as an interface diagnostic; do not call it a final benchmark.
4. Train on agreed split with checkpoint/resume support for preemptible embers.
5. Run fixed online evaluations: propose 20 rollouts for debugging, then 50 for a reported single-task result; freeze initial-state IDs/seeds/horizon before comparison. These counts are proposals, not executed trials.
6. Report successes/total and uncertainty, videos, latency methodology and settings. Determine a reasonable numeric target from a matching implementation/protocol or explain absence of one; no invented paper-equivalent threshold.
7. Only after a credible ACT policy, proceed to other baselines and then A2A.

## Resource scope
No new dependencies or weights installed/downloaded by source inspection. No ACT GPU jobs submitted.
Use free embers only and exclude prior ECC node; keep simulation environment stable.
Longer model runs need a concrete compute/time/storage plan before submission.
