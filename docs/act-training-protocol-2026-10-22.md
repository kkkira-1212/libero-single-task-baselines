# ACT single-task experiment — 2026-10-22

Goal: move beyond the 200-update/0-of-2 pilot to a meaningful training/evaluation result. Completion of this budget does not automatically mean a reproduced reasonable success rate.

## Fixed data and model
Task: LIBERO-Object alphabet soup to basket, same verified HDF5 and 40/10 whole-demo split.
Seed0; 9D joint/gripper state, two128x128 cameras, 7D delta OSC actions, chunk20, author ACT CVAE/Transformer.
Train-only normalization; original recorded same-index observation/action convention retained.
Pretrained ResNet18 weights from https://download.pytorch.org/models/resnet18-f37072fd.pth, 46830571 bytes, SHA256 f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec. Cached in scratch torch-cache, not Home.
Changes relative to yesterday: pretrained backbone instead of random; 10000 rather than200 updates; batch16 rather than8; main LR1e-5 rather than5e-5. This is a new run, not continuation of pilot optimizer state.

## Training
Config configs/act/train_seed0.json; runner experiments/act_pilot.py with training_only=true.
10000 updates, batch16, main/backbone LR1e-5, KL10, grad clip1. No temporal ensembling.
Validation criterion: deterministic prior inference valid-action L1 on160 fixed timesteps spanning all10 held-out demos. Model selection uses minimum sampled validation loss, never online evaluation success.
Checkpoint every500 updates: latest full model/optimizer/RNG/config/history; separate best-policy model-only export for fast evaluation.
User resource scope: one free embers GPU,4CPUs,24GB RAM,max2h, ~3–4GB scratch for checkpoints and atomic writes plus cached weights. No paid account use.
Stop new updates after6600 seconds process time to reserve checkpoint time; incomplete target returns nonzero, so dependent evaluation will not silently run as if training completed. If preempted, preserve latest and review before resubmission; automatic resume is not yet configured.

## Evaluation
Separate dependent job requires successful training (afterok). Only one evaluation allocation, free embers,max45min,1GPU/4CPU/16GB RAM.
20 fixed initial-state IDs0..19/seeds0..19,10 settling steps,600 policy steps cap, replan every5 steps from20 predicted actions, no temporal ensemble.
600-step horizon follows official LIBERO libero/configs/eval/default.yaml (not yesterday's150-step diagnostic).
Save per-rollout success/steps/actions/GIFs, completed count and Wilson95% interval. Partial/cancelled trials reported explicitly. Single training seed and20 evaluations limit precision; no suite-level or paper-equivalence claim.
Reasonable success-rate numeric target still needs a genuinely matched reference; do not invent one. If performance remains poor, inspect behaviors and training diagnostics before scaling.

## Submitted jobs and evidence
Training14293363: logs/act-train-14293363.log, artifacts/act-pilot/14293363/result.json, scratch act-pilot/14293363/latest.pt and best-policy.pt.
Evaluation14293364 (afterok14293363): logs/act-eval-trained-14293364.log, artifacts/act-eval/14293364/result.json.
BEGIN/END/FAIL email enabled. Evaluation still needs resources after dependency releases; completion today is not guaranteed by submitting jobs.
