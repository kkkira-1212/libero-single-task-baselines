# DP execution interval diagnostic

### Final DP execution-interval comparison and BAKU model gate passed
- DPexec1evaluation14353581 COMPLETED0:0 in1h33m13s:16/20=80%,Wilson95%58.40–91.93%,all20trials/GIFs/actionfiles complete.
- Frozen samecheckpoint exec5control14353583:10/20=50%,36m39s,sameIDs20..39/seeds/GPUmodel. Observed+30percentagepoints;not yet robust generalization.
- Paired improvements(exec5failure->exec1success):[24, 25, 26, 30, 34, 36, 37, 38]. Regressions:[27, 31]. ID38 now successful per simulator;inspect new video before causal interpretation.
-8improved/2regressed;exploratoryexacttwo-sidedMcNemarp=0.109375. Singlecheckpoint/seed,reuseddevelopmentstates,noise-streamdifferencesafterfirstreplan;fixedintervalcomparison,notadaptivepolicyvalidation.
- Totalwalltimeabout2.54xincludesstartup/simulation/rendering/inference/I/O;notstandaloneinferencelatency.
- BAKUsetup14354834 COMPLETED0:0 in6m20s;modelcheck14354835 COMPLETED0:0 in8m32s onA10080GB. AuthorBCAgent realbatchupdatepassed,loss14.51063,finitegradients/parameters,actorparametermaxchange0.0001000166;7Dactionandsave/perturb/reloadmaxerror0.0.
- BAKUhistory1/query10,oneupdateonly;live_environment_steps=0,online_rollouts=0. NextrealheadlessLIBEROintegrationbeforetraining.
- No new jobs submitted during this review.
