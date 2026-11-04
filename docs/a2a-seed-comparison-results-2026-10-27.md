# Frozen A2A seed diagnosis results

No training updates. Same selected expert inputs,unchanged checkpoints. Training GPU and seed were not independently controlled.

Full raw/EMA comparison complete:True. CPU preliminary complete:True. Oracle complete:True.
CPU preliminary covers32train/16validation windows only;fullcomparison has broader prespecified phase coverage.

Actual training source snapshots are identical;both40000drawsequences were reconstructed with exact SHA matches.
Seed0/1coverage99.756%/99.838%;gripper-switchdraws538/530. Seed1sawall86switchanchors.

| Weight | Split | Seed | Windows | AE first translation RMSE | Flow6 first translation RMSE | Flow6 first grip-sign accuracy |
|---|---|---|---|---|---|---|
| policy | train | 0 | 238 | 0.066044 | 0.103476 | 0.958 |
| policy | validation | 0 | 64 | 0.070235 | 0.142209 | 0.9688 |
| ema_policy | train | 0 | 238 | 0.066845 | 0.105824 | 0.958 |
| ema_policy | validation | 0 | 64 | 0.070871 | 0.142953 | 0.9688 |
| policy | train | 1 | 238 | 0.063487 | 0.106689 | 0.9496 |
| policy | validation | 1 | 64 | 0.068226 | 0.149303 | 0.9531 |
| ema_policy | train | 1 | 238 | 0.06397 | 0.108319 | 0.9538 |
| ema_policy | validation | 1 | 64 | 0.068634 | 0.149698 | 0.9688 |

Errors above are in raw controller-command input units,not measured robot-position error.
Latent coordinate bases can differ across trained models;compare decoded actions rather than swapping cross-seed encoders/decoders.

| Demonstration | Mode | Seed | Success | Steps | Matched GT succeeds |
|---|---|---|---|---|---|
| demo_28 | GT | None | True | 126 | True |
| demo_28 | ae | 0 | True | 125 | True |
| demo_28 | flow6 | 0 | False | 136 | True |
| demo_28 | ae | 1 | True | 125 | True |
| demo_28 | flow6 | 1 | False | 136 | True |
| demo_36 | GT | None | True | 140 | True |
| demo_36 | ae | 0 | False | 151 | True |
| demo_36 | flow6 | 0 | True | 140 | True |
| demo_36 | ae | 1 | False | 151 | True |
| demo_36 | flow6 | 1 | False | 151 | True |

AE oracle uses future ground-truth actions;flow oracle uses prerecorded expert observations. These are diagnostic teacher-forced replays,not deployable-policy success or benchmark results.
Two demonstration replays do not establish all-task/component correctness. No unique failure cause is automatically inferred.
Closed-loop contact sheet:artifacts/a2a-seed-comparison/static/closed_loop_frames.png.
Machine-readable phase metrics:artifacts/a2a-seed-comparison/summary/result.json.
