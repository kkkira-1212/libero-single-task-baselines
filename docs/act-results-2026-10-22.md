# ACT initial-budget results — 2026-10-22

## Verified result
Training14293363 completed10000updates in35m47s (pretrained ResNet18,seed0,40training demos).
Validation-selected checkpoint is step1000, not step10000. Minimum prior valid-action L1 on160 held-out timesteps0.469946; final step10000 metric0.483949.
Evaluation14293364 completed all20 fixed rollouts,600-step cap,5-step execution from20-step chunks, no temporal ensemble:
- successes2/20 =10%.
- Wilson95% interval2.79%–30.10% across these trials; one training seed, no seed-to-seed variability estimate.
- Successful initial states0 (430steps) and19 (288steps); other18 reached600steps unsuccessfully.
- Full evaluation job elapsed17m22s, not inference latency.
- GIFs/trajectories/report artifacts/act-eval/14293364; success_failure_preview.png shows two successes and a failure.

## Interpretation and limitations
Visually inspected examples show learned grasping/transport; sampled failure0/1 sequence carries the can near the basket but does not finish placement. One failure example is not a failure taxonomy.
This is real task success but weak reliability; do not declare mentor's reasonable-success baseline gate complete.
Later sampled training reconstruction errors decreased, while held-out prior inference metric plateaued; these use different samples/conditioning and do not prove overfitting or explain failures on their own.
The offline selection metric may not rank online policies well. Compare the saved final10000-update model on the SAME20 initial states before deciding to train longer.
These20 initial states become a development/model-comparison set once reused for selection; a final unbiased result should use a separately declared available initial-state set and/or independently repeated seeds. Do not call adaptively reused trials a held-out final test.
Pilot0/2 at150steps is not directly comparable to2/20 at600steps with changed training settings.

## Follow-up
Submitted14296407 to evaluate final10000-step checkpoint (latest.pt) with exactly the same20-state600-step protocol, free embers,max45min,no new updates.
If final checkpoint improves online behavior, reconsider checkpoint selection. Otherwise inspect repeated failure modes and test one controlled change at a time (execution/temporal ensemble, data alignment or training recipe), not blind scaling.
No claim of complete ACT reproduction yet.

## Final10000-update checkpoint result
Comparison14296407 completed all20 rollouts:12/20=60% (Wilson95%38.66%–78.12%), versus2/20=10% for step1000 on the same development states/seeds/protocol.
This is a50-percentage-point observed improvement. Offline sampled L1 selected the weaker online policy in this comparison. Evaluation ran on different GPU types; repeated seeds/hardware controls were not performed.
The step10000 checkpoint is the current candidate for independently fixed confirmation trials. This development result establishes useful task behavior, not paper-level reproduction or a final untouched test score.

## Frozen confirmation protocol
Frozen candidate: step10000, byte-identical read-only copy in scratch frozen/act_seed0_step10000/checkpoint.pt; SHA256564e098e7fee352626743ac54bc5e808388041c3779c8747eb111d80a37a0784.
Protocol configs/act/confirmation_frozen.json fixed before outcomes:20new initial IDs20..39,seeds20..39,all action/horizon settings unchanged. Abort if state IDs unavailable; no substitution.
Job14297140 submitted, free embers,max45min. Results pending. These IDs are new to development evaluation, not established as disjoint from training demonstrations. No claim of multiple-training-seed robustness.

## Frozen confirmation outcome
Job14297140 completed all20 prespecified new-state trials with13successes (65%, Wilson95%43.29%–81.88%). Frozen SHA256 verified, zero training updates, V100,600step cap,unchanged chunk/action schedule.
Development12/20 (60%) and confirmation13/20 (65%) support useful performance beyond checkpoint selection states. Report separately. Descriptive combined25/40 (62.5%) mixes selection and confirmation and is not an unbiased40-trial final test.
This is a working single-task/single-training-seed ACT baseline candidate, not suite-level/paper-score reproduction. Freeze it for subsequent method comparisons; no further tuning on this confirmation set.
