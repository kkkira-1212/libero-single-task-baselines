# A2A implementation audit — 2026-10-26

Scope: inspect pinned author source a5792ecf4e7f8fa4d85fe66ea9a50618138f925c, paper v2, and our actual LIBERO runner/checkpoints. No new training and no silent model/loss modifications.

## Findings with evidence and confidence

| Item | Confirmed evidence | Interpretation and next test |
|---|---|---|
| Source/target physical spaces | Official il_run.sh sets joint_pos/joint_pos; data2zarr_dp.py uses measured joint_qpos and desired joint_qpos_target. Our source is joint7+gripper2; target is OSC delta pose6+binary gripper1. | Confirmed adaptation difference. Width9->7 patch fixes tensor compatibility, not the same-space continuity prior. High-priority design risk, NOT proved cause of all0/20. |
| Action execution schedule | DefaultEvalRunner uses DiffusionPolicyCfg(temporal_agg=False), n_action_steps=8; BaseEvalRunner caches and executes the8 returned actions while updating history every real step. Our runner replans after1step. | Confirmed execution difference. Frozen2-state exec1vs8 probe isolates it without retraining. |
| Observation/action time | LIBERO exporter stores post-action obs with action at same index; this was documented and preserved for all our baselines. Current A2A uses obs[t] ending after command[t], labels future action slice starting at command[t]. | Confirmed timing convention; differs from current measured-state->next-target interpretation. Do not silently shift only one benchmark method. Need explicit causal alignment validation or separate corrected data version. |
| Binary gripper | All6196 training action rows have gripper -1/+1 only; measured finger positions continuous. | Continuous latent decoder may blur switching times; official README also flags switch-like dimensions. Dense per-horizon transition error and oracle AE replay directly test bottleneck. |
| Missing pretraining? | Official DefaultRunner.train runs joint model.compute_loss/backward/optimizer/EMA, with no obligatory separate AE-pretraining stage on this base route. | No evidence that an omitted required stage explains failure; previous suggestion was a hypothesis, now deprioritized. |
| Paper/code latent consistency | Paper v2 Eq5 uses latent L1; pinned a2a_policy.py:198 uses MSE. Our runner uses pinned author MSE, not a substituted loss. | Confirmed paper/repo gap; do not silently change. Numeric component audit first, separately labeled ablation only if justified. |
| YAML fields | action_ae use_variational,kl_weight,freeze_encoder/freeze_decoder,recon_loss_type are present but base policy/CNN encoder do not implement all these flags. | Config presence is not mechanism. No evidence we forgot an enabled VAE stage. |
| Image preprocessing | Author RobotImageDataset uses image range normalizer2x-1 and encoder ImageNet normalization. Our adapter intentionally keeps [0,1] before encoder-owned ImageNet normalization. | Confirmed difference; our train/live path intended to be consistent with encoder contract. Not established to be a bug or reason to copy upstream double transform blindly. |
| Optimization | SameAdamWlr/betas/weightdecay,warmup500,cosine,EMA.75; batch4 instead of author32; extra gradclip1; 40/10 split rather than author default2%validation; samples chosen timestep-uniform. | Experiment is an adaptation, not identical paper recipe. Additional updates cannot alone validate batch/schedule equivalence. |
| Torch requirement correction | Selected A2AImagePolicy constructs SimpleFlowNet (MLP). SDPA appears in unused Attention/FlowTransformer classes in the same file. | Earlier claim 'selectedA2A requiresTorch2 becauseSDPA' was too strong. CurrentTorch2env works; no reason to reinstall for this audit. |

## Evidence files
- adapters/a2a_model.py:5–38, experiments/a2a_run.py:19–116 andlive/evalloop.
- third_party/A2A/roboverse_learn/il/il_run.sh:obs_space/act_space.
- third_party/A2A/roboverse_learn/il/data2zarr_dp.py:163–235.
- third_party/A2A/roboverse_learn/il/policies/a2a/a2a_policy.py:98–132,141–223.
- third_party/A2A/roboverse_learn/il/runners/default_eval_runner.py:30–55; base_eval_runner.py:get_action.
- third_party/A2A/roboverse_learn/il/runners/default_runner.py:train; configs/train_config/default_train.yaml.
- artifacts/a2a-audit/static/data_semantics.json.
- [Paper v2 §3.2–3.3](https://arxiv.org/html/2602.07322v2#S3).
- [Author deployment notes](https://github.com/JIAjindou/A2A_Flow_Matching#real-robot-deployment).

## Runtime audit — zero parameter updates
CPU14467555: reconstruct author total loss numerically, inspect every major module gradient, compare offline/live8-frame prediction on same history, permute image/history to measure sensitivity, dense action AE errors by horizon and gripper switch frames. Save oracle actions.
GPU14467556(afterokCPU): same-demo original action replay vs future-action-AE oracle reconstruction on train demo28/valdemo36; then2-state exec1/8 probe capped300steps. Oracle uses future ground truth and is NOT deployable policy or benchmark success rate. AE failure only interpretable if matched ground-truth replay succeeds. State-only initial restore without native XML is a known replay limitation.
The initial all-GPU audit14467439 was still pending and was replaced by this split to avoid GPU queue blocking mathematical investigation.

## Decision rule
1. Original replay fails -> first resolve replay/controller/state validity; do not blameAE.
2. Original replay succeeds and AE oracle fails -> demonstrated representation/execution bottleneck beforeflow.
3. AE oracle succeeds but predicted policy fails -> focus on history-to-target mapping, visual conditioning and closed-loop alignment.
4. Exec8 rescues frozen policy -> investigate execution-schedule mismatch before retraining.
No single confirmed discrepancy above is claimed to prove the cause of all failures; wait for controlled tests.

No model parameters, source loss formula or dataset were changed during audit.

## Completed CPU evidence (job14467555)
- COMPLETED0:0,4m59walltime;themathematicalbodytook121.65s. No parameterupdates.
- Hand-reconstructed loss exactlymatchedauthorcompute_loss:0.0217543617on4fixedwindows.
- Components:flowMSE0.000347803;AE_L1.017623667;latentconsistencyMSE.000292477;latentL1on SAMEtensors.013381799;flowactionL1.024604496. Thisquantifiespaper/codeformula-scalegap,notaprovedfix.
- All6majorblocksreceivednonzero finitegradients:visualencoder.02114,visualprojector.003819,historyencoder.17445,futureencoder.08158,decoder.29356,flownet.74950 L2norms. Differentmodulenormsarenotdirectlearnabilityscores.
- Same8framehistory offlineadaptervs onlinehistorybuilderpredictionmaxdifference0.0. Thisrulesoutamismatchonthatwindow,notarbitrarysimulatorstatehistory.
- Imagepermutationchangedactionsmeanabs.01798;historyreversal.08737on4windows. Modelusesbothsignals,butthesearenotcausalperformanceeffects.
- DenseAEfirsttranslationRMSEdemo28.03832/demo36.03748rawcommandunits;interiorchunkpositionsroughly.019–.027. First-actiongrippersignaccuracy98.53%/98.68%;bothsampleddemos'2switchframeswerecorrect. Noevidenceforuniversalwronggrippersign;rareamplitude/timingerrorsstillpossible.
- PendingGPUoracle/execprobe14467556. Added CPUphysics-onlyoracle14467705:ControlEnvwithcamera/offscreenrenderdisabled,originalvsAEactions,no neuralnetworktraining/inference. ThiscananswerreplayquestionwithoutGPUqueue. Saveeef/grippertrajectories andstrictlyconditionAEconclusiononGTsuccess.

## Completed isolation results and current conclusion
All audits used the SAME frozen40kseed0EMA checkpoint SHA b2082dbf6f2778408f0ac6b6c7569550f180271289c378085c8d6fe02b17f9e7;zerooptimizerupdates.

| Case | Original action replay | Ground-truth-future action AE reconstruction replay |
|---|---|---|
| Train demo28 | success126steps | success132steps |
| Validation demo36 | success140steps | success129steps |

CPU14467705 andGPU14467556 agreed on these matched demonstrations. These are TWO demonstrations repeated in two execution environments, NOTeight independent benchmark trials. AEoracle uses future truth;it is not an executable policy. Result shows the decoder/controller path can execute successful actions inthese cases;it does NOTprove perfectAE or generalization.

Frozen full policy probe oninit20/38,exec1vs8,capped300steps:allfourfailed. Thisdoesnotruleoutfull600step/other-stateeffects,butdoesnotjustifytreatingexec8asanalreadyverifiedfix.

Official-sampler numeric test (33commonwindows) passed. Ourfuture slice[7:15]matchespinnedcode;ourloaderaddsoneverylastcurrenttimestepperdemo whereasupstream16/7paddingendsatL-2. Thisminorboundarydifferenceisnot evidenceforan8-frameindexingbug.

**Narrowed failure location:** predicting correct future actions from history/vision and maintaining it in closed loop. We have evidence against a completely broken decoder/controller, disconnected gradients, missing base-loss terms, or an identical-window normalization mismatch as a universal explanation. We do NOThave proof of a unique cause.

**Highest-priority unresolved adaptation assumptions:** position-feedbacksource vsdelta-controltarget,post-action-observation labels vs truly futurecommand alignment,andclosed-loop history/start-state distribution. Their existence isconfirmed;their causal contribution to0/20must be tested independently. Paper/codeL1-vsMSE andimagepreprocessingdifferences are fidelity issues,not automatic fixes.

## Concrete next gate (no budget scaling)
1. Preserve alloldcheckpoints/results and label them cross-space/post-action-convention adaptation.
2. Write an explicit causal sample contract: current state is after command[t], next target command starts at[t+1]. Validate on restored states/controller replay. If adopting this version for comparison, apply/document a common corrected data protocol across methods;never silently shift onlyA2A.
3. Specify source and target physical representations with units/frame/controller scaling. Compare a same-space executed-EE-history design or absolute-target design only after inverse controller mapping is verified. Do not treat9->7tensor support as a validated physical prior.
4. Use a fixed tiny demonstration segment to test the specific representation/alignment hypothesis, with strict gradient and frozen-decoder checks, before any new longtraining.
5. Only after controlled evidence supports a change, run one explicitly labeled corrected adaptation;no uncontrolledloss/seed/budget grid.

Sources: [paper v2 §3](https://arxiv.org/html/2602.07322v2#S3),pinnedsourcefilesabove,andartifacts/a2a-offline-audit/14467555/result.json,artifacts/a2a-oracle-replay/14467705/result.json,artifacts/a2a-audit/14467556/result.json. Examplefailedpolicyframes:artifacts/a2a-audit/static/failure_frames.png.


## Follow-up: causal timing and controller-goal verification completed
CPU job14467957 passed with zero training updates. All50episodes matched post-step proprioception to next raw pre-state exactly;250explicit next-command windows passed. Actual OSC goals invert back to commands within8.9e-9, but measured executed translation differs from commanded displacement by pooledXYZRMSE~0.016m on two successful GTdemonstration replays. Thus measured motion is not an exact normalized command surrogate. See [versioned contract and detailed results](libero-causal-physical-contract.md). This is not yet a learned-policy improvement or a unique-cause attribution. Next isolate causal timing before changing the physical source representation.


### 2026-10-27 — paired timing correction result verified:0/20 ->20/20
- Allsixchainjobscompleted0:0;no Slurm jobsremainqueued/runningatinspection. Controltrain14468637/eval14468638:0/20,all600stepcap. Causaltrain14468639/eval14468640:20/20,136..141steps,mean138.15.
- Same seed0,10kupdates,batch4,6156train/1602validationanchors,split,normalizer,architecture,loss,finalEMA,6flowsteps/exec1,initialIDs20..39andseeds,600stepcap. Allfour train/eval allocationsRTX6000. Configdifferencesonlysample_contractanddescriptivepurpose.
- ActualtrainingsampleorderSHAidentical:6ecafec239e6832089d01bfda15a90734dbebe8d5deca7c5eaf1d1d3304e4a43. GPUgatepreviouslyverifiedidenticalinitialstateSHA andactualencoderlabels. Thisstronglysupportstimingmismatchasakeyfailurecauseinthisprespecifiedpairedsetting,notuniqueglobalcauseproof.
- Verifiedall40completedrolloutrecords,40GIFheaders/nonemptyfiles,40actionfiles7D/clipped/lengthmatchesrecord;checkpointsandconfigsagree. No new simulation or training necessary for result verification.
- IndependentstreamedSHAofthecausalcheckpointmatchedreportedadec7729806f88631e201c25fb2317e4bcaea0e317a4b4bca543b337c1e770f2. Frozenmanifestconfigs/a2a/frozen_causal_seed0_10000.json preservespath/hash/config/protocol;oldfailedcontrol/checkpointsretained.
- Fixed32windowvalidationtotal_losscontrol0.07209104vscausal0.06967620;smalldifferenceinlossdoesnotpredictlargeclosedloopsuccessdifference.
- Thisestablishesaneffectivesingle-task6flowstepA2Acandidate,not1flowstep/latency/suite-levelortrainingseedrobustness. Nextrecommendedconfirmations:independenttrainingseedandfreshinitialstatesbeforebroadclaims,thenfrozen1flowstepeval. No additionaljobs submittedduringthisinspection.


### A2A frozen inference hardware check completed — discrepancy follows checkpoint
- TypedRTX6000retry14503459 COMPLETED0:0 in18m22s;collector14503460 COMPLETED0:0. SlurmAllocTRESgpu:rtx_6000=1andtorchQuadroRTX6000confirmrequiredhardware;20completeevaltrialswithGIF/action/config/SHAchecks.
- Frozen seed0:RTX6000reference20/20,V10014498425=20/20. Frozen seed1:V100reference0/20,RTX600014503459=0/20. Weights unchanged,6flowsteps/exec1/IDs20..39/seeds20..39,600stepcap.
- Outcome follows the two frozen checkpoints across tested inference GPUs. Inference GPUtypealone doesnotexplainthis20/20vs0/20gap. TrainingGPUandtrainingseedwerechangedtogether,so their training-time contributions remainunseparated.
- Nextdiagnosticfocusisfrozentraining-checkpoint/data-fit/AE-vs-flow/closed-loopbehaviorcomparison,not moretrainingupdatesorblanketdeviceblame. No newtraining submitted duringresultinspection.
