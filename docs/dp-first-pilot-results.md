# DP first single-task pilot results

Training job14350906 completed3000updates in22m07s; evaluation14350907 completed20rollouts in41m23s. Both QuadroRTX6000/freeembers,ExitCode0:0.

- Final predetermined3000-update EMA policy:10/20 successes=50%,Wilson95%29.93–70.07%.
- Fixed initial IDs/seeds20..39,600steps,10settle steps,execute5thenreplan,100DDPMsamplingsteps.
- All20GIFs and20action files present. Ten failures reached600steps; failure mechanism requires video inspection.
- Data-loader boundary parity and EMA reload placement checks passed.
- Same task and40/10demo split asACT. DP3000updates/random-init imagebackbones versusACT10000updates/pretrainedbackbones; architecture,normalization,history andtrainingrecipes differ. These are preliminary adapted policies, not a controlled architecture-only ranking.
- ACTseed0:13/20=65%;ACTseed1:9/20=45% on sameIDs. DPonlyone trainingseed. Do not claim DP superior/inferior from these20trials.
- Validation noise-MSE is not ACTL1 and cannot be compared numerically across methods. Limited fixed32validation samples do not establish online success.
- Evaluation walltime includes simulation,inference,rendering,serialization andstartup. It is not a standalone latency benchmark.
- Checkpoint ~/scratch/robot_manipulation/dp-training/14350906/latest.pt, SHA25610c9b963fe52dbbd7e08b2e099811371cc1926b4c806ecbbce3a79d0f96424b1.
- Evidence artifacts/dp-eval/14350907/result.json; training metrics artifacts/dp-train/14350906/metrics.json.
- No new experiments submitted during result review. Next inspect successful/failed trajectories, preserve candidate, then choose replication/longertraining orBAKU preparation with user.

### Qualitative review of saved DP frames
- Reviewed evenly sampled frames from one successful reference(ID21) and all10 failures; produced artifacts/dp-eval/14350907/review_all_failures.png and review_selected.png.
- Selected close-up rows:21(success),20,25,38. Original GIFs retained. GIF samples occur about every5simulation steps; final image is replaced by exact terminal frame. Enlarging128px frames adds no detail.
- ID21 visibly lifts the target and transports it to the basket; simulator reports success after147steps. Final release is not established by sparse images.
- ID20 spends late sampled frames near the tabletop/object region without completing transport. ID25 later moves toward the basket without a clearly visible carried target; missed/unstable grasp is a hypothesis, not confirmed by contact/object-state measurements.
- ID38 visibly lifts/transports the target, then target appears outside/right of basket in later frames; simulator reports failure at600steps. This supports a distinct transport/placement-stage issue, not a universal failure to grasp.
- Sparse low-resolution views cannot establish exact contact, release timing or root causes. Current evaluator saved executed actions and external camera GIFs, not object pose/contact/state or wrist-camera trajectories; future instrumented diagnostics would be needed to confirm mechanisms.
- No failure-category counts assigned and no hyperparameter changes/new jobs submitted from visual inspection alone.
