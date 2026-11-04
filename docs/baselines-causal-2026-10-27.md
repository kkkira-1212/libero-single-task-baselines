# Causal-v1 ACT, Diffusion Policy and BAKU experiments

Same task,40train/10validation demos,obs[t] -> next command[t+1],IDs20..39,settle10,600stepcap.
Each method retains its prior budget/model/optimizer/normalizer/padding/sampling/execution recipe.
This is not equal-compute or original suite-level reproduction. Historical numbers are not a paired timing ablation.

| Method | Training seed | Updates | Batch | Execution | Training job | Eval job | Result |
|---|---|---|---|---|---|---|---|
| act | 0 | 10000 | 16 | 5 | 14497118 | 14497119 | 17/20 |
| act | 1 | 10000 | 16 | 5 | 14497120 | 14497121 | 17/20 |
| dp | 0 | 3000 | 16 | 5 | 14497124 | 14497125 | 15/20 |
| dp | 1 | 3000 | 16 | 5 | 14497127 | 14497129 | 12/20 |
| baku | 0 | 10000 | 16 | temporal aggregation | 14497138 | 14497139 | 20/20 |
| baku | 1 | 10000 | 16 | temporal aggregation | 14497140 | 14497141 | 6/20 |

DP exec1 is a prespecified secondary frozen-weight evaluation, not a best-result selection.

- DP seed0, exec1, eval14497126: 19/20
- DP seed1, exec1, eval14497132: 19/20

Incomplete trials are not scored as failures. Failed gate jobs cannot release dependent training.
ACT uses final10k checkpoint, DP final3kEMA, BAKU final10k; both training seeds must be reported.
A2A causal seed0 reference20/20 at6flowsteps and19/20 at1step;training batch4/10k/exec1 differs from these recipes. Its seed1 confirmation remains in docs/a2a-confirmation-2026-10-27.md.
