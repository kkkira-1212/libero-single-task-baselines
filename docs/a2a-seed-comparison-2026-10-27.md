# Frozen A2A seed comparison — 2026-10-27

Question: both training configurations actively changed only seed (plus a descriptive purpose string), so why do two trained checkpoints give20/20vs0/20?

## Confirmed provenance and sampling
- Saved training source snapshots match byte-for-byte for a2a_run.py,a2a_model.py,a2a_policy.py,libero_sequence.py,libero_causal_contract.py.
- Dataset/split/architecture/loss/normalizer recipe/10kupdates/batch4/learning-rate schedule are the same. TrainingGPU differed (RTX6000vsV100),so this is not a strict same-hardware seed-only intervention.
- Reconstructed actual NumPy random draws exactly:SHA256matches each run's recorded sample-order hash. Both drew40000anchors from6156eligible anchors.
- Seed0:6141uniqueanchors(99.756%),538gripper-switch draws,85of86available switchanchors seen,positivecommand fraction0.541175.
- Seed1:6146uniqueanchors(99.838%),530gripper-switch draws,86of86switchanchors seen,positivecommand fraction0.5446.
- Only8outof40000drawpositions contained the same anchor in both sequences. Similar coverage does not make optimizer trajectories identical.
- This does not support a simple explanation that seed1 missed gripper-switch training examples.

Evidence:artifacts/a2a-seed-comparison/static/source_and_sampling.json.

## Closed-loop evidence inspected
Same initialstates20and38,compare frames at0,40,80,120andterminal.
Seed0 succeeds140/136steps;seed1fails600/600steps. In both cases trajectory differences appear during initial approach,with different motion directions and gripper timing.
ForID20,first30action meanY is-0.418(seed0)vs+0.331(seed1);first positive gripper command occurs at54vs0. ForID38,Y is-0.433vs+0.349;first positive grip52vs0.
These are descriptive differences,not a unique underlying-cause attribution. No claim that early positive grip alone causes failure.

Contact sheet:artifacts/a2a-seed-comparison/static/closed_loop_frames.png.
Action summary:artifacts/a2a-seed-comparison/static/closed_loop_summary.json.

## Zero-update model diagnosis
- Full raw/EMA,AE/flow1/flow6 comparison:GPUjob14513136. Identical deterministic windows across both frozen checkpoints,including beginning/end/middle and gripper-switch frames;actual current causal-v1 adapter is used.
- CPU preliminary actual EMA-weight comparison:14513419,32train+16validation windows selected from the same prespecified plan. This is a preliminary subset,not a fullsplit error estimate.
- Matched CPU physics-only oracle replay:14513226,afterok14513136. Native demonstrations28(train)/36(validation);GTbaseline once each,then AEfuture-truth and expert-observation flow sequences for bothseeds. No camera/rendering/model training.
- All explicit freeembers. FullGPU1GPU4CPU24GBmax30min;CPUpreliminary4CPU16GBmax30min;physics2CPU16GBmax45min. BEGIN/END/FAILemail enabled.
- Source,split,configuration,SHA/target slicing and manual-author loss parity checked. Compare normalizerstats;unscaled first-action translation/rotation/gripper errors;AE vs Flow;raw vs EMA;within-checkpoint latent statistics.
- No optimizer/backward/weight updates in these jobs. Learned latent coordinate bases differ across independently trained models:do not swap encoders/decoders across seeds or infer failure from raw latent coordinate differences alone.
- Oracle AE uses future ground-truth commands;flow oracle uses recorded expert observations rather than live feedback. Both are diagnostic oracles,not deployable policies. Retain GTcommand0 because the first stored observation is post-command0;only compare modelreplays when matched GTreplay succeeds.

Current status:static provenance/sampling/video checks complete;component metrics and oracle outcomes pending.


### CPU preliminary frozen comparison completed — same experts,similar errors,closed-loop differs
- CPU14513419 COMPLETED,zero parameterupdates;bothEMAcheckpoints evaluatedonidentical32train/16validationexpertwindows. Config/split/checkpointSHA/manual-authorlossparity/predict-decompositionchecks passed. Inputnormalizerparametersidentical(maxabsdifference0.0).
- FirsttranslationcommandRMSE:trainAE0.05725(seed0)vs0.05670(seed1),flow6=0.10342vs0.10965;validationAE0.07914vs0.07745,flow6=0.18102vs0.18529. Errors in raw controller-command units,not physical position errors. Firstgrippersignaccuracyidentical:31/32train,15/16validation.
- Subsampledinitialrepeatedhistoryandgripswitchwindowmetricsalsoclose;no clear evidence thatseed1'sAEisgloballybroken. Thisis48selectedexpertwindows,notalltrainingdataorproofthattheflowisalwayscorrect.
- Initial20/38closedloopsexhibitdifferentfirstmotions/gripdespitethesecloseexpertinputerrors. Hypothesistoinvestigate:behavioronactualonlineinitial/off-expertstates;unique cause NOT proven.
- FullGPU14513136stillPrioritypending;nowalsoincludesbothEMAmodelsreadingthesameactualbenchmarkinitialobservations(ID20/38)anddiagnosticinputhybridswithouttraining. Hybridsarenotphysicalrolloutsuccessoruniquecauseproof. Maincomponentarrayoutputsremainavailableevenifoptionalrenderprobeerrors.
- Physicsoracle14513226afterokfullcomparison;collector14513611waitsfull/CPUprelim/physics. Reports docs/a2a-seed-comparison-results-2026-10-27.md and artifacts/a2a-seed-comparison/summary/result.json;plan docs/a2a-seed-comparison-2026-10-27.md.
