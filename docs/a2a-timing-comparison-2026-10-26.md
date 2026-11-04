# A2A timing-only paired experiment

Single LIBERO task, seed0, 10000 updates, 6 flow steps, execute1; fixed IDs20..39.
Only target-label timing differs; both variants exclude the same terminal observation anchors.
No paper reproduction, multi-seed robustness or causal physical-prior improvement is claimed.

Data check passed: True. GPU gate status: passed. Paired actual sample order verified: True.

| Variant | Training job | Evaluation job | Completed rollouts | Success |
|---|---|---|---|---|
| control | 14468637 | 14468638 | 20 | 0/20 |
| causal | 14468639 | 14468640 | 20 | 20/20 |

Partial runs are not scored as zero. A numerical paired comparison requires both complete evaluations and matching actual training sample-order digests.

Manifest: configs/a2a/timing_comparison_2026-10-26.json; raw reports remain under artifacts/a2a-{train,eval}/<job>.
