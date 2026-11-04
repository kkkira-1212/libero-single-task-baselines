# A2A causal-v1 confirmation experiments

Reference: frozen seed0 model, 6 flow steps, IDs20..39, 20/20 successes.
All evaluations use execution horizon1 and600stepcap. Different questions remain separate; no pooling.

| Experiment | Job | Flow steps | Expected rollouts | Result |
|---|---|---|---|---|
| seed0_newstates_steps6 | 14495966 | 6 | 10 | 10/10 |
| seed0_steps1 | 14495967 | 1 | 20 | 19/20 |
| seed1_steps6 | 14495969 | 6 | 20 | 0/20 |

No latency is measured in these experiments. New initial states are within the same task.
Existing ACT/DP/BAKU results use the historical data convention; a corrected-protocol cross-method ranking requires a consistent timing audit/adaptation.
Plan: exam Thursday Oct29; aim to consolidate Sunday Nov1 and report Monday Nov2.
