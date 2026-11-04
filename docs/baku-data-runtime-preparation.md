# BAKU initial data-interface runtime preparation

### 2026-10-24 — follow-up inspection and BAKU data-interface jobs
- User authorized continued code explanation and queueing interface preparation alongside DP experiments.
- At inspection14353584 COMPLETED0:0 in6m39s onA10080GB,all3000updates,seed1.14353585 evaluation had just started;14353581(exec1),14353583(exec5) still running. No final follow-up success rate yet. Exec5 partial IDs20..23 included3successes; do not report partial rate as final.
- Added isolated BAKU data preparation environment scripts and pinned dependency requirements,usingPython3.8.19/Torch1.11 for this initial data/embedding gate. This differs from upstream Python3.9 recipe; full model/simulation environment not yet validated. ACT/DP envs untouched.
- CPU prepare14353908 installs minimaldata/MiniLM dependencies in scratch/baku-env, resolves then pins actual HuggingFace MiniLM commit in saved report, computes genuine384Dembedding for task instruction,verifies datasetSHA,exports40train/10validationPKLs via existingexporter. Cache/outputs in personal scratch. Expected fewGB disk;2CPU/16GB/max1h,freeembers,emailenabled.
- Dependent CPU audit14353909(afterok14353908) checks whole-demo split,all action/proprio values,first/last dual-camera vertical flips,then exercises unchanged official BAKU BCDataset on bothsubsets. Expected batch2/history2/query20:RGB[2,2,3,128,128],proprio[2,2,9],actions[2,2,20,7],text[2,384]. Report exactshapes,finitevalues,PKLhashes.
- This is a bundled data/author-loader gate,NOT BAKU model forward/backward/checkpoint/live rollout or formal training. Model head imports include robomimic/VQ dependencies; full import audit and GPU model integration remain next.
- PythonAST andbashsyntax passed before submissions;runtime unverified. Reports artifacts/baku-prepare/14353908/result.json and artifacts/baku-data-check/14353909/result.json; logs/baku-prepare-14353908.log and logs/baku-data-check-14353909.log.
- Official BAKU loader uses its own nearidentity preprocessing(x/1.00001),episode/window sampling,zero-padded future chunks. Preserve/report these differences from DP limitsnormalization/repeated-boundary padding rather than silently unifying algorithms.

### BAKU export failure diagnosed and retried
-14353908 FAILED1:0 after6m51s. Dependency installation andpipcheck passed; realMiniLM weights downloaded and384Dembedding generated. No BAKU model updates/rollouts.
- Exporter incorrectly indexed h5py dataset with negative vertical stride ([:,::-1]); h5py raisesValueError:Step must be>=1. This is our exporter bug,not missingGPU/dependencies.
- Fixed both camera reads to first load NumPy arrays ([:]),then vertically reverse rows andmakecontiguous. Preserves intended orientation/timing; original data unchanged.
- Added early model revision provenance save before export,so downstream failures preserve source identification.
- Canceled old afterok audit14353909 because predecessor failed; submitted replacement prepare14354211 and afterok audit14354213,existing freeCPU resource limits/email unchanged. Existing packages/cache reused. End-to-end runtime fix awaits results.
- DP14353581/14353583/14353585 stillrunning at failure review; do not restart unaffected jobs.

### BAKU data gate passed; DP exec5 repeat completed
- BAKU prepare14354211 COMPLETED0:0 in4m55s; audit14354213 COMPLETED0:0 in3m19s. H5 negative-stride fix verified end-to-end.
- Genuine MiniLM embedding generated from revision1110a243fdf4706b3f48f1d95db1a4f5529b4d41; dataset checksum preserved. Exported40train/10validation demos in scratch/baku-data/14354211.
- Official BCDataset yielded expected finite batch shapes on BOTHsubsets; actions/proprio values and boundary-camera vertical flips verified. Model forward/backward=false,online_rollouts=0: BAKU training is still pending model integration.
- DP exec5 control14353583 COMPLETED0:0 in36m39s:10/20=50%,Wilson95%29.93–70.07%. All20 pertrial success/step counts exactly match original14350907,confirming repeat of frozen evaluator; NOT20additional independent initial states.
- DPexec1 job14353581 stillRUNNING:7completed trials atinspection. Seed1eval14353585 stillRUNNING:19completed trials. No final rate or method/timing conclusions from partial results.

### DP seed1 final result; BAKU model integration queued
- DPseed1eval14353585 COMPLETED0:0 in28m59s,9/20=45%,Wilson95%25.82–65.79%,all20IDs20..39completed. Seed0sameprotocol10/20=50%; retain both training seeds,insufficient evidence for stable method ranking.
- DPexec1evaluation14353581 stillRUNNING,10/20completed atinspection; no final interval conclusion.
- Submitted BAKU model dependency/import CPUjob14354834 (2CPU/16GB/max1h) and afterok GPUcombinedcheck14354835 (1GPU/4CPU/24GB/max30min),explicitfreeembers,knownbadECCnodeexcluded,GPUemailenabled.
- Existing BAKUenv receives robomimic0.2.0/diffusers0.11.1 for upstream eagerimports; ACT/DPenvs unchanged. No algorithm source modifications.
- Modelcheck uses unchanged officialBCAgent,ResNet+FiLM+GPT,deterministichead,language/proprio,hidden256,officialLIBERO historyfalse/query10/temporalaggregationtrue (earlier data gatehistory2/query20 was a loader test only).
- Check2realtraining samples,one officialupdate,finiteparameters/gradients per module,actorweight change,7Daction on storedimage/state,author snapshot save/perturb/reload and deterministic action agreement. Snapshot in scratch;config/sourcecommit incheckpoint.
- This job explicitly has live_environment_steps=0 andonline_rollouts=0: runtime simulation integration remains a subsequent gate,not silently claimed complete. Current check/training runtime unverified.
- Pre-submit PythonAST/bashsyntaxpassed; logs/baku-model-setup-14354834.log andlogs/baku-model-check-14354835.log; resultartifacts/baku-model-check/14354835/result.json.

### BAKU headless live integration prepared and submitted
- User authorized continuation to live LIBERO gate after model integration passed.
- Added adapters/baku_live.py:rawuint8 dual-camera images verticallyflip thenHWC->CHW;state=joint7+gripper2;official nearidentity datasetstats passed toauthoract;noDP/ACTnormalization reuse. Modelact owns imageToTensor.
- CPUsetup14373867 adds MuJoCo2.3.7/robosuite1.4/LIBERO andcompatiblepinned simulationdependencies toisolatedbaku-env,2CPU/16GB/max1h. NumPy aligned toknownLIBERO1.22.4 indata/simrequirements;this changesBAKUenvonly,pipcheck+imports/freeze run beforeafterokGPU.
- GPUlivecheck14373868 afteroksetup,1GPU/4CPU/24GB/max30min,explicitfreeembers,knownECCnodeexcluded,BEGIN/END/FAILemail.
- Check loads1-update checkpoint14354835 withauthorBCAgent;restoresdemo28states[1],compares9Dliveproprioagainststoredobs[0]at1e-6 andflippedimageMAE diagnostic(<15),matchingdocumentedpostactiontiming.
- Then resetsfixedinit20,10settlezeros,andexecutes30closedloopstepsviaauthoract/temporalaggregation;checks7Dfiniteactions,clip[-1,1],finitesimulation,savesGIF/wristPNG/actiontrajectory/report.
- No newtraining;online_rollouts=0,training_updates_this_run=0. This is30stepintegration,not success-rate evaluation. Model isnotyet trained for task.
- AST/bashsyntaxpassed;runtimechecksawaitjobs. Logs/baku-sim-setup-14373867.log,logs/baku-live-check-14373868.log;artifacts/baku-live-check/14373868/result.json,live.gif,actions.json,wrist_final.png.

### BAKU live LIBERO integration passed
- Simulation setup14373867 COMPLETED0:0 in6m51s;pipcheck/importspassed,BAKUenvNumPy1.22.4 withMuJoCo2.3.7/robosuite1.4.0. ACT/DPenvspreserved.
- GPUcheck14373868 COMPLETED0:0 in6m44s,onQuadroRTX6000/freeembers:passed=true,30closed-loopsteps,finite7Dactions/simulation.
- Demo28restoredstate[1]vsstoredobs[0]9Dpropriomaxerror0.0. Flippeddual-cameraMAE3.0342external/2.0595wrist,consistentwithdocumentedrenderdifferences.
- Verified30actionrecords,live.gif,wrist_final.png,result.json present underartifacts/baku-live-check/14373868/.
- Used1-update interfacecheckpoint14354835;training_updates_this_run=0,online_rollouts=0. This establishes realobservation->officialBAKUact->simulation chain,nottaskperformance.
- BAKUdata/model/checkpoint/liveinterfacegatesnowpassed;nextpreparefixedsingle-tasktrainingrecipeanddependent20trialclosed-loopevaluation. No trainingjobs submitted during thisresultreview.
