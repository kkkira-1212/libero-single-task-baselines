# Causal-v1 results overview — 2026-10-27

Single LIBERO-Object task0;40training/10validation demonstrations;next-command labeling;fixed evaluation initialIDs20..39,seeds20..39;20rollouts/model;600stepcap.

| Method / evaluation recipe | Updates | Training seed0 | Training seed1 |
|---|---|---|---|
| ACT,chunk20,execute5 | 10000 | 17/20=85% | 17/20=85% |
| DP,DDPM100,execute5(primary) | 3000 | 15/20=75% | 12/20=60% |
| DP,DDPM100,execute1(prespecified secondary) | same frozen3kweights | 19/20=95% | 19/20=95% |
| BAKU,query10,temporal aggregation | 10000 | 20/20=100% | 6/20=30% |
| A2A,6flowsteps,execute1 | 10000 | 20/20=100% | 0/20=0% |

All eight ACT/DP/BAKU evaluations completed0:0,with config/source/checkpoint/trialaction/GIFheader checks. No pending primary baseline evaluations.

## Interpretation
- ACT's two runs agree at85%;DPexecute1reaches95%inbothruns. This is evidence on one task and two trained models,not a general benchmark ranking.
- DPexecute1andexecute5reuseidenticalcheckpointSHApertrainingseed. Seed0has5improvedinitialstatesand1regression(ID32);seed1has7improvementsand0regressions. Both execute1runsfailonlyID32.
- DPexecute1meansreplan after each simulator action;itstilluses100DDPMdenoisingsteps. This is different from A2Aone-flow-stepgeneration.
- BAKU20/20vs6/20andA2A20/20vs0/20show substantial cross-run variation. Do not report only best seeds. Training/evaluationGPUtypesalsovary,so seed alone is not isolated as a cause.
- The methods retain different update budgets,batches,histories,padding,sampling,pretraining and execution recipes. Common timing/split/task does not make this an equal-compute experiment or suite-level reproduction.

## A2A additional checks
Frozen seed0:6stepnewinitialIDs40..49=10/10;1flowstepIDs20..39=19/20.
Frozen seed0movedfromRTX6000toV100andkept6steps:20/20in14498425. Thus V100doesnotgenerallypreventthismodel/taskfromworking.

The intended failed-seed1RTX6000check14498426 actually executed onV100(confirmedtorchGPUname,SlurmAllocTRESandlogs),giving0/20again. This is a V100repeat,NOTthe intended RTX6000comparison. Preserve that evidence;do not claim both hardware directions were completed.

Retry14503459requiresgpu:rtx_6000:1andfeatureRTX6000,with a runtime model-name assertion before simulation. It has no training updates and keeps weights/SHA/6steps/exec1/20initialstates fixed. Collector14503460validates actual GPU type. Verified result:seed1onQuadroRTX6000remained0/20;SlurmtypedGRESandGPU-nameguardpassed. Both frozen inference-hardware directions are now complete. Inference hardware tests do not isolate training GPU from training seed.

## Evidence
- Main baseline table:docs/baselines-causal-2026-10-27.md
- A2Aconfirmation:docs/a2a-confirmation-2026-10-27.md
- A2Ahardwarecheck:docs/a2a-hardware-check-2026-10-27.md
- Machine-readable outputs underartifacts/baselines-causal/2026-10-27andartifacts/a2a-confirmation/2026-10-27.


### A2A frozen inference hardware check completed — discrepancy follows checkpoint
- TypedRTX6000retry14503459 COMPLETED0:0 in18m22s;collector14503460 COMPLETED0:0. SlurmAllocTRESgpu:rtx_6000=1andtorchQuadroRTX6000confirmrequiredhardware;20completeevaltrialswithGIF/action/config/SHAchecks.
- Frozen seed0:RTX6000reference20/20,V10014498425=20/20. Frozen seed1:V100reference0/20,RTX600014503459=0/20. Weights unchanged,6flowsteps/exec1/IDs20..39/seeds20..39,600stepcap.
- Outcome follows the two frozen checkpoints across tested inference GPUs. Inference GPUtypealone doesnotexplainthis20/20vs0/20gap. TrainingGPUandtrainingseedwerechangedtogether,so their training-time contributions remainunseparated.
- Nextdiagnosticfocusisfrozentraining-checkpoint/data-fit/AE-vs-flow/closed-loopbehaviorcomparison,not moretrainingupdatesorblanketdeviceblame. No newtraining submitted duringresultinspection.


### 2026-10-28 morning — open-gripper initialization confirmation completed20/20forbothfrozenmodels
- Confirmation14516670 COMPLETED0:0in10m33s,zero trainingupdates,bothmodelssequentiallyonthesameTeslaV100-PCIE32GB. EachprespecifiedIDs20..39/seeds20..39,6flowsteps/exec1,10settleactions,600stepcap.
- Explicitopen-gripperwarmup(grip=-1,arm0):seed0=20/20,126..134steps(mean129.5);seed1=20/20,120..134steps(mean128.2). All40initialgripcommandsnegative(open).
- Verified40completedrolloutrecords,40GIFheaders,40actionfiles(length/finite7D/clipping),originalcheckpointconfigs/10ksteps/SHAandactualGPU. Weightsunchanged,no training.
- Originalall-zerowarmupresultsremainseed0=20/20,seed1=0/20onbothinferenceGPUtypes. Pairedchanged-initializationinterventionsupportsinitialgripperstateasakeytriggerforthesefrozenmodels/onthissingletask;itdoesnotestablishcanonicalprotocolrobustnessoruniquetrainingfailurecause.
- Trainingexpertinitialgrip~(+.0362,-.0362),originaleval~(+.0205,-.0205);expertgrip-onlyinputsubstitutionflipsseed1openwherearm-onlydoesnot. Realopenwarmupand40rolloutconfirmationextendthetwo-stateinputprobe.
- Today'smeetingmustpresentoriginalanddiagnosticconditionsseparately. Candidatefollow-updiscussion:retainreferenceinitializationandimproveinitial-state/historycoverageorrobustness;anynewcommoninitializationprotocolmustbeexplicitandappliedconsistently,beforecomparisons.
- MorningPPToutlineupdatedwithcompletedcounts;no jobsrunningatinspection. No newexperiments submittedinthisresultcheck.
