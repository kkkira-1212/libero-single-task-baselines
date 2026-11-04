# Next-method preparation — 2026-10-22

## Actual readiness
Official sources pinned and inspected; shared temporal data windows tested on real data. No Diffusion Policy, BAKU or A2A dependencies installed, GPU forward/backward run or training/evaluation launched. These methods are NOT yet turnkey training-ready. The remaining setup gates below must be completed before saying they can train tomorrow.

| Method | Author source commit | Prepared | Remaining gate |
|---|---|---|---|
| Diffusion Policy | 5ba07ac6661db573af695b419a7947ecb704690f | source, image-policy/encoder audit, 2-frame history adapter, shape config | isolated dependencies, official normalizer/model factory, combined forward/backward + rollout smoke |
| BAKU | 47e367059191de30ac4b6353ff5615fdef11d8c5 | official LIBERO path, single-task Hydra config, split-aware PKL exporter | isolated dependencies/MiniLM weights, actual conversion, task override, combined smoke |
| A2A | a5792ecf4e7f8fa4d85fe66ea9a50618138f925c | policy audit, 8-frame history adapter, dimension mismatch identified | separate 9D history encoder vs 7D action AE; dependency/import audit, then experiments only after baselines |

## Reusable data/evaluation contract
- Same task and 40/10 whole-demo split/checksum as ACT. No validation data used to fit normalizers.
- State is 7 joint positions + 2 gripper positions; action is 7D delta OSC pose + gripper.
- adapters/libero_sequence.py emits raw [0,1] CHW images and raw state/action values, with repeated boundary samples. Each method applies its own explicitly recorded normalization, exactly once.
- Current action is at history index n_obs_steps-1; horizon16/history2 (DP) or history8 (A2A). Both tested at two episode boundaries and first/last indices. Do NOT pass returned metadata/padding keys blindly to an upstream policy; construct the required obs/action dictionaries.
- Existing HDF5 uses post-action obs and pre-action states. Preserve this documented convention across method comparisons or deliberately create and verify a common corrected dataset for ALL methods, never change just one.
- ACT first-run online protocol: fixed initial-state IDs0..19, seeds0..19, max600 policy steps, 10 settling steps.600 matches official LIBERO default eval config. Method-specific chunk/replan/ensemble choices are separate recorded settings.

## Diffusion Policy
Use original diffusion_policy/policy/diffusion_unet_image_policy.py and ConditionalUnet1D, not a reimplementation.
MultiImageObsEncoder expects RGB in [0,1] and optionally handles ImageNet normalization. Do not feed ACT's already normalized images into it.
Upstream full conda environment targets Python3.9, torch1.12.1, diffusers0.11.1; also installs a custom robosuite/mujoco-py stack. Do not install that file into the working LIBERO environment. Start with an isolated minimal policy environment and an explicit LIBERO dependency plan.
Normalizer API and diffusion scheduler/EMA settings must be traced and fixed before runtime checks.
Prepared config: configs/diffusion_policy/single_task.json.

## BAKU
Official Instructions.md already supports LIBERO, train.py/eval.py, agent=baku suite=libero dataloader=libero. No from-scratch model needed.
Single-task override template: configs/baku/libero_object_single.yaml (copy into chosen Hydra config path before use).
Official suite uses joint+gripper9D and vertically flips BOTH online cameras. read_data/libero.py reads per-task PKL and a fixed MiniLM text embedding.
scripts/export_baku_single_task.py exports the existing train/validation demo IDs, requires a genuine precomputed384D MiniLM embedding (no fake zeros). Syntax checked only; conversion not run.
Upstream generator rerenders states[t] rather than using stored obs[t], changing temporal alignment. Our proposed exporter deliberately preserves the ACT data time convention and records that difference.
Original environment requests Python3.9/Hydra1.1/einops0.7, sentence-transformers, dm_env/dm_control; import/dependency smoke needed in an isolated environment.
The repository also has MT-ACT, useful as a LIBERO reference/backup; do not silently substitute it for our recorded ACT implementation.

## A2A
Source confirms history source uses normalized obs["agent_pos"], visual conditioning and future action encoders/decoder. Current history_action_encoder is constructed with action_dim, assuming history-state width equals action width. Our9D state vs7D action needs separate parameters, just as ACT did.
History CNN needs at least8 frames. Initial padding must stay within episode.
Do not assume the full RoboVerse/Isaac environment is necessary to use the policy; audit imports and adapt to existing LIBERO/MuJoCo where feasible.
No published latency or original task score is transferable to our setup without measurement.
Prepared requirements description: configs/a2a/libero_adapter_requirements.json.

## Priority while ACT runs
1. Finish isolated dependency validation and DP model factory.
2. Obtain/cache MiniLM embedding and verify BAKU conversion + task override.
3. Bundle each method's data/model/loss/checkpoint/rollout checks into one job.
4. Begin substantive DP/BAKU runs after ACT results and common protocol are reviewed; A2A experiments remain gated on the baselines.

## 2026-10-23 runtime update
DP isolated dependency setup14339568 and afterok combined model/live integration14339569 submitted; no successful imports/model-runtime claims until results. Adapter/factory and checks now implemented (docs/dp-runtime-preparation.md).
ACT independent seed1 train/eval14339488/14339489 runs alongside this preparation. BAKU/A2A runtime still unvalidated.
