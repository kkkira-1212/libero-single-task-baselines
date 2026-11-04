# Integrated ACT training/evaluation pilot

Purpose: consolidate prior interface checks into a bounded training-to-rollout pipeline; not final baseline evaluation.
Implementation: experiments/act_pilot.py; config configs/act/pilot.json; Slurm scripts/act_pilot.sbatch.

## Pilot protocol
- Original 40/10 whole-demo split and training-only normalization reused.
- Randomly sample training timesteps across all 40 training demos, target 200 updates, batch 8, main LR 5e-5 / backbone LR 1e-5, KL 10, clip 1, chunk 20. Random ResNet18 initialization; no weight downloads.
- Validation prior(z=0) L1: 4 fixed timesteps per held-out demo (40 samples total), masked and normalized by valid elements. This is sampled validation, not full-set loss.
- Atomic checkpoint every 50 updates, plus final; saves model, optimizer, Python/NumPy/Torch/CUDA RNG, config, split, normalization and history to scratch.
- At first checkpoint: reference update -> restore optimizer/model/RNG -> replay same next update; compare sampled indices and all model state tensors. Replay tolerances rtol 1e-5, atol 1e-6.
- --resume PATH restores state with strict config/split/normalization/interface checks. Cross-job resubmission/preemption recovery remains untested until actually exercised.
- Node-local temporary dataset copy avoids repeated random access over scratch; size checked, original dataset not modified.
- Before rollout: load stored simulation state, compare normalized proprioception, compare raw/flipped image MAE to stored images. Abort rollout if mapping checks fail.
- Two diagnostic rollouts only: initial states 0,1, seeds 0,1, 10 zero-action settling steps, 150 policy steps maximum, control 20 Hz, replan every 5 steps from 20-step chunk, no temporal ensembling.
- Save actual clipped actions, rewards, done, success flags, GIFs. These are pipeline diagnostics, not sufficient evidence for a reasonable benchmark success rate.
- No latency/throughput claim from total job duration; initial imports, dataset staging and checkpoint IO included.

## Resource and monitoring
Free embers, account gts-agarg35, 1 GPU/4 CPUs/16 GB RAM, max 30 minutes.
Excluded prior ECC node. Full optimizer checkpoints ~1 GB, atomic replacement can temporarily double occupancy; allow ~2–3 GB scratch.
Target stop adding training steps after 20 minutes process time; reserve time for final checkpoint and evaluation. SIGUSR1/SIGTERM sets cooperative-stop flag; shell forwards signals. This is best effort and does not guarantee a checkpoint survives abrupt cancellation.
No new dependencies, weights or datasets downloaded.
Current submitted job 14274392: logs/act-pilot-14274392.log; artifacts/act-pilot/14274392/result.json and history.json; checkpoint in ~/scratch/robot_manipulation/act-pilot/14274392/latest.pt.
Rerun only after checking queue; do not duplicate active jobs.

## Observed timing convention and retry
First pilot 14274392 completed 200 updates with held-out sampled L1 1.03133 -> 0.53409 and exact next-update recovery, then stopped before rollouts due to an incorrect parity probe.
Official dataset generation stores post-action obs[t], pre-action states[t], and action[t]. Numeric verification on two training episodes showed obs[t] proprioception equals states[t+1] exactly. The parity probe now restores states[1] when checking obs[0]. Training target indexing remains consistent with the official same-index loader; no data shift applied.
Retry 14275022 restores existing step200 checkpoint, so no further training is intended. Expected runtime <=15 minutes; verify updates_this_run=0 in output. Pilot rollout counts/horizons remain diagnostics.

## Retry outcome and dedicated evaluator
Retry14275022 restored the checkpoint with zero training updates and passed proprioception parity exactly. Camera orientation also passed. It reached the Slurm advance-warning signal before rollout, so rollouts=[] despite original execution_ok=true. This was an incorrect completion-status predicate; it has been corrected. Do not interpret the original job exit0 as a completed policy test.
Dedicated evaluator experiments/act_eval.py avoids optimizer setup/restore, repeated validation and checkpoint writes. It returns success only after all requested episodes complete or terminate normally. Job14276098 uses30min free embers, unchanged checkpoint/rollout protocol, no training.
