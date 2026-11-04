# Frozen A2A inference-hardware cross-check

Seed0 train/eval RTX6000:20/20. Seed1 train/eval V100:0/20.
These jobs move each fixed checkpoint to the other inference GPU type,with unchanged6flowsteps/exec1/IDs20..39/seeds20..39.

| Case | Job | Reference | Cross-hardware result |
|---|---|---|---|
| seed0_v100 | 14498425 | 20/20 | 20/20 |
| seed1_rtx6000 | 14503459 | 0/20 | 0/20 |

Previous attempt14498426 executed on V100 despite the requested RTX6000 partition; retained as a V100 repeat, not the intended RTX6000 check.

No training updates. This checks inference hardware effects; it does not isolate training hardware from training seed.
