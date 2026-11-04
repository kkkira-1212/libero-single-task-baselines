# A2A two-checkpoint diagnosis findings — 2026-10-27

## Completed evidence
GPU comparison14513136completed0:0in7m58s,ononeV100forbothmodels. It used238train/64validationchosen expertwindows(total302)acrossall50demos,notalltimesteps. BothrawandEMAweightswerecompared withzero trainingupdates. CPUoracle14513226completed0:0in5m20s;summary14513611completed.

Training sources/split/normalizer recipe match. All four raw/EMA normalizerstate dictionaries are identical(maxabs0). Actual40000trainingdrawsperseedwereverifiedbySHA;coverage99.756%/99.838%,grip-switchdraws538/530. Seed1sawall86switchanchors. Thus we have not found a missing-data or mismatched-normalizer explanation.

## Expert-window results
| EMA metric,64validationwindows | seed0 | seed1 |
|---|---|---|
| AE first translation-command RMSE | .07087 | .06863 |
| Flow6 first translation-command RMSE | .14295 | .14970 |
| AE first grip-sign accuracy | 100% | 100% |
| Flow6 first grip-sign accuracy | 96.875% | 96.875% |

Units are raw controller input command units,not robot-position errors. Raw-weight results are similarly close. This is not proof that EMA has no closed-loop effect:raw-weight closed-loop success was not measured here.

Initial expert windows:40train+10validation;bothseeds'firstgripper-signaccuracy100%. Expert gripper-switch windows:train86both100%,validation24both91.67%. Average and selected phase errors alone do not explain20/20vs0/20closed-loop outcomes.

## Same real initial observation: most discriminating current clue
The two EMA models received exactly the same rendered images and repeated8-frame proprioceptive history on the sameGPU. Their first executed actions match previously saved benchmark first actions within2.4e-7,so the probe reproduces the known behavior.

PandaGripper.format_action uses the sign of the gripper command:-1opens,+1closes. Magnitude is not proportional opening force.

| Input condition,initialID20 | seed0 gripper command | seed1 gripper command |
|---|---|---|
| Same real online observation | -.478(open) | +.849(close) |
| Expert images + real proprioception | -.461(open) | +.778(close) |
| Real images + nearest expert initial proprioception | -1.000(open) | -1.000(open) |

ID38hasthesamequalitativepattern:live-.576vs+.869;expertimages/liveproprio-.417vs+.720;liveimages/expertproprioboth-1.000.
All real initial proprioception coordinates remain inside training min/max bounds;nearest expert initial-state distance is~1.11in model-normalized9Dcoordinates. Being inside individual coordinate ranges does not establish coverage of the joint state/history combination.

These input-hybrid examples are off-manifold one-step sensitivity checks on two states. They point to sensitivity to initial robot state/history,not a unique cause or proof that vision is unimportant. They do not distinguish the7armjointsfrom2gripperfeedbackvalues.

## Oracle replays: diagnostic limits
Native GT commands succeed on train demo28andvalidationdemo36.
AEfuture-ground-truth replay:bothseeds succeed on28(125steps),bothfailwithin151commands on36.
Flow with prerecorded expert observations:bothfailwithin136commands on28;seed0succeeds36(140steps),seed1failswithin151commands.
These replay sequences have a fixed demonstration length,not the600stepbenchmark horizon. Future-GT AE and prerecorded-observation flow are teacher-forced diagnostics,not deployable-policy scores. This does not establish a globally broken decoder unique to seed1.

## Next controlled zero-update test
Job14515931isolates onlyarmvsgripperfeedbackinfirst-actionprobes,andrecordsactualjoint/gripperpositionsandnormalized distances. It also physically compares10warmupsteps with originalzero gripper command against explicitopen(-1),forbothfrozenEMAmodels onIDs20/38,300stepcap(8shortrollouts). Raw-weight first-action predictions are checked too;no newraw-weightbenchmark score is implied.

Important: pinned official LIBERO metric.py uses5all-zero dummyactions;our frozen commonprotocol uses10all-zero steps. Explicitopenwarmupis a deliberate diagnostic condition,NOTanestablishedbenchmarkcorrection. Do not replace the original20/20vs0/20primaryresults with scores from changedinitialization.

Frozen models remain unchanged. No additional training. The trainingGPU-vs-trainingseedconfound is still separate from the now-completed inferenceGPUcheck.


### 2026-10-28 morning — open-gripper initialization confirmation completed20/20forbothfrozenmodels
- Confirmation14516670 COMPLETED0:0in10m33s,zero trainingupdates,bothmodelssequentiallyonthesameTeslaV100-PCIE32GB. EachprespecifiedIDs20..39/seeds20..39,6flowsteps/exec1,10settleactions,600stepcap.
- Explicitopen-gripperwarmup(grip=-1,arm0):seed0=20/20,126..134steps(mean129.5);seed1=20/20,120..134steps(mean128.2). All40initialgripcommandsnegative(open).
- Verified40completedrolloutrecords,40GIFheaders,40actionfiles(length/finite7D/clipping),originalcheckpointconfigs/10ksteps/SHAandactualGPU. Weightsunchanged,no training.
- Originalall-zerowarmupresultsremainseed0=20/20,seed1=0/20onbothinferenceGPUtypes. Pairedchanged-initializationinterventionsupportsinitialgripperstateasakeytriggerforthesefrozenmodels/onthissingletask;itdoesnotestablishcanonicalprotocolrobustnessoruniquetrainingfailurecause.
- Trainingexpertinitialgrip~(+.0362,-.0362),originaleval~(+.0205,-.0205);expertgrip-onlyinputsubstitutionflipsseed1openwherearm-onlydoesnot. Realopenwarmupand40rolloutconfirmationextendthetwo-stateinputprobe.
- Today'smeetingmustpresentoriginalanddiagnosticconditionsseparately. Candidatefollow-updiscussion:retainreferenceinitializationandimproveinitial-state/historycoverageorrobustness;anynewcommoninitializationprotocolmustbeexplicitandappliedconsistently,beforecomparisons.
- MorningPPToutlineupdatedwithcompletedcounts;no jobsrunningatinspection. No newexperiments submittedinthisresultcheck.
