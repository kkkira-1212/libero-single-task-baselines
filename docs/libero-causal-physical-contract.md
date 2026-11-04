# LIBERO causal timing and physical action contract — diagnostic v1

Status: verification job 14467957 passed; no model training and no old training adapter switched.

## Time indexing

Let u[t] be stored action[t]. LIBERO's exporter executes u[t] and then saves obs[t], so obs[t] describes the state after u[t]. Raw states[t] is the state before u[t]. A deployable next-action sample must therefore use:

- Observation history: obs[t-7], ..., obs[t].
- Future command targets: u[t+1], ..., u[t+8].
- Repeat only within the same episode at boundaries; expose masks for repeated entries.
- Reject t = episode_length-1 because no real next-command label exists.

The opt-in implementation is adapters/libero_causal_contract.py. It also provides versioned full-window packing for the opt-in training loader. Historical configurations retain their old default behavior. The old A2A window selects targets u[t], ..., u[t+7]. Comparisons that change this convention must be explicitly labeled; a corrected common-protocol comparison requires applying the timing convention consistently to the baselines.

The exported dataset lacks an image before its first stored action. This contract does not invent one; action[0] is not a next-command target conditioned on a real preceding stored observation.

## Three different signals

1. Normalized OSC command: translation3 + rotation3 in [-1,1], plus a separate gripper command.
2. Controller goal: a desired world position and orientation generated from the current pose and scaled command.
3. Executed motion: measured end-effector displacement and orientation change after simulation.

These signals cannot be treated as identical. The controller sets R_goal = Exp(delta_rotation_world) @ R_current. Reconstruct a relative rotation with Log(R_goal @ R_current.T); subtracting absolute axis-angle vectors is not a valid general rotation inverse. An exactly zero orientation command retains the previous goal, which needs explicit handling.

The audit intercepts actual set_goal calls without changing commands, checks forward and inverse scaling, then measures how much executed motion differs from commanded displacement. The same replay also checks that the instrumentation still permits successful demonstration execution.

Measured gripper position is continuous feedback; binary open/close is a command. This audit does not claim an invertible mapping between them.

## Decision gate

A passing audit verifies a causal index contract and controller coordinates/scales. It does not prove that either issue explains the failed A2A policy. First isolate a timing-only controlled experiment while preserving the existing model architecture, split and prescribed evaluation protocol. A proposed physical-history source needs a separate explicit representation and normalization design; do not simultaneously change time indexing, physical prior, loss and training budget.

Evidence will be stored in artifacts/a2a-causal-physics/14467957/result.json and per-demonstration trajectories.


## Verified results — job 14467957

Free embers CPU-only job completed in 5m02s, 2 CPUs, requested 16GB; measured peak memory about 2.24GiB. No rendering, GPU, optimizer or parameter update.

- Dataset SHA256 42189d4415d4c51aaaf0708300653fccc39239cd3f2709079a713cd8d1678a8d.
- All 50 episodes: stored joint/gripper obs[t] matches raw pre-state[t+1] exactly (maximum absolute difference 0). All 250 representative causal-index windows passed, including boundary masks and rejecting missing next commands. Held-out episodes were inspected for schema/timing only; no fitting or normalization used them.
- Native train demo28 replay succeeded in 126 steps; validation demo36 in 140 steps. This checks replay/instrumentation, not learned-policy performance.
- Actual OSC scaling: normalized translation multiplied by 0.05 meters per coordinate; normalized rotation multiplied by 0.5 radians in axis-angle form. No position/orientation goal limits active.
- Forward position-goal error 0; rotation-matrix max error below 5.8e-9. Inverse normalized command error below 8.9e-9 for all position commands and nonzero rotation commands. Exactly-zero rotation held the previous goal on 7/6 steps, checked separately.
- Measured motion vs commanded displacement pooled XYZ RMSE: 0.01607/0.01572 meters (normalized equivalent 0.32146/0.31448). These are differences between single-step displacement signals, not absolute end-effector trajectory-position errors. Normalized rotation-signal RMSE 0.02433/0.02582. Results cover only these two demonstration replays.

The test confirms the next-command labeling correction and disproves a simple identity between measured EE displacement and normalized OSC commands in these trajectories. It does not prove either issue uniquely caused the policy's 0/20 score. Do not replace the A2A prior with measured deltas under an assumption of exact command equivalence.

Next controlled intervention: causal-timing-only A2A pilot with the old model/physical source/loss/split retained; use an explicitly versioned run and prescribed evaluation conditions. Keep old results for provenance; comparisons with baselines require a common timing convention. No new training was submitted as part of this verification.


## Timing-only correction integrated and queued

The author model is unchanged: its future action encoder still reads packed action slots [7:15]. The LIBERO adapter now separates observation indices from action indices:

| Variant | Observation history | Packed 16 action slots | Actual 8 future targets |
|---|---|---|---|
| Matched control | obs[t-7:t+1] | action[t-7:t+9] | action[t:t+8] |
| Causal v1 | obs[t-7:t+1] | action[t-6:t+10] | action[t+1:t+9] |

Slices use Python's exclusive upper bound; indices are clipped within their own episode. Action slots before7 and slot15 are unused by the author A2A loss. The unchanged model returns these8decoded actions in order, so online evaluation executes output0 immediately; no further inference-side shift is added.

Both paired variants use the same anchors t=0..L-2, excluding the last observation of each episode. This gives6156train/1602validation anchors, keeps40/10split, train-only normalization, seed0, batch4, AdamW, EMA,10kupdates, architecture, losses, images and proprioception unchanged. Loss padding remains endpoint repetition; validity masks do not introduce a new masked loss. The comparison is with a freshly paired control because historical runs included the last anchor; comparing only against historical0/20would add a sample-population difference.

The source before modification is archived under artifacts/a2a-causal-change/source-before/. Changes are in adapters/libero_causal_contract.py, adapters/libero_sequence.py and experiments/a2a_run.py. Old configs without sample_contract still use same-index targets and all anchors. New configs explicitly declare sample_contract and sample_anchor_policy. Each training run snapshots the adapter sources and records the actual sampled-anchor order SHA256.

Prespecified jobs: data check14468711; GPU integration14468575; control train14468637->eval14468638; causal train14468639->eval14468640; collector14468641. Each train waits for successful GPU integration and verifies the tested source hashes; each eval waits for successful training. Training checkpoint selection is final10kEMA, no online selection. Evaluation is6flowsteps/exec1,600stepcap,initialIDs20..39andmatchingseeds. No1step,physical-source,loss,architecture or extra-seed sweep is part of this intervention.

Current runtime evidence is pending; submission and syntax checks alone do not establish that the correction improves learned performance. Results: docs/a2a-timing-comparison-2026-10-26.md; manifest configs/a2a/timing_comparison_2026-10-26.json.


### Actual loader validation passed
CPUdata job14468711passed:200windowsacross40train episodes, paired observations exactly equal, both target slices checked against raw HDF5, cached/reference loader equality, old default behavior preserved. Dataset SHA unchanged;6156train/1602validationanchors;zero parameterupdates. The GPUloss/update/reloadgate14468575isstillqueued,so both training chains have not started at this milestone.

Decision rule: if both matched runs remain0/20, timing correction is not a sufficient effective-policy fix at10kupdates; do not automatically increase training budget. Any improvement is a single-seed exploratory result, not unique root-cause proof or a comparable corrected-protocol baseline suite.


### 2026-10-27 — paired timing correction result verified:0/20 ->20/20
- Allsixchainjobscompleted0:0;no Slurm jobsremainqueued/runningatinspection. Controltrain14468637/eval14468638:0/20,all600stepcap. Causaltrain14468639/eval14468640:20/20,136..141steps,mean138.15.
- Same seed0,10kupdates,batch4,6156train/1602validationanchors,split,normalizer,architecture,loss,finalEMA,6flowsteps/exec1,initialIDs20..39andseeds,600stepcap. Allfour train/eval allocationsRTX6000. Configdifferencesonlysample_contractanddescriptivepurpose.
- ActualtrainingsampleorderSHAidentical:6ecafec239e6832089d01bfda15a90734dbebe8d5deca7c5eaf1d1d3304e4a43. GPUgatepreviouslyverifiedidenticalinitialstateSHA andactualencoderlabels. Thisstronglysupportstimingmismatchasakeyfailurecauseinthisprespecifiedpairedsetting,notuniqueglobalcauseproof.
- Verifiedall40completedrolloutrecords,40GIFheaders/nonemptyfiles,40actionfiles7D/clipped/lengthmatchesrecord;checkpointsandconfigsagree. No new simulation or training necessary for result verification.
- IndependentstreamedSHAofthecausalcheckpointmatchedreportedadec7729806f88631e201c25fb2317e4bcaea0e317a4b4bca543b337c1e770f2. Frozenmanifestconfigs/a2a/frozen_causal_seed0_10000.json preservespath/hash/config/protocol;oldfailedcontrol/checkpointsretained.
- Fixed32windowvalidationtotal_losscontrol0.07209104vscausal0.06967620;smalldifferenceinlossdoesnotpredictlargeclosedloopsuccessdifference.
- Thisestablishesaneffectivesingle-task6flowstepA2Acandidate,not1flowstep/latency/suite-levelortrainingseedrobustness. Nextrecommendedconfirmations:independenttrainingseedandfreshinitialstatesbeforebroadclaims,thenfrozen1flowstepeval. No additionaljobs submittedduringthisinspection.
