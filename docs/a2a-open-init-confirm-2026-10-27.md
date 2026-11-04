# A2A open-gripper initialization confirmation

Changed diagnostic initialization,not a replacement for the original all-zero warmup scores.
Same frozen10kEMAweights,IDs20..39/seeds20..39,6flowsteps,execute1,settle10,horizon600. No training.
Both checkpoints are evaluated on the same allocatedGPU.

| Training seed | Success | Job / artifact |
|---|---|---|
| 0 | 20/20 | 14516670/seed0 |
| 1 | 20/20 | 14516670/seed1 |

Original zero-warmup results remain seed0=20/20,seed1=0/20.
Compare the full conditions and report remaining training-GPU/seed confounding and single-task scope.
