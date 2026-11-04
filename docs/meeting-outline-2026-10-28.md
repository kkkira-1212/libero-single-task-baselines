# Meeting outline — 2026-10-28

Meeting preparation date: October28(America/New_York). Final6-slide deck and832-word English script are ready in reports/2026-10-28/. Aim for8–10minutes plus discussion. Exact meeting time was not specified. ThursdayOctober29exam remains a constraint. User requested90minutes total preparation,including at least30minutes paper review.

Purpose:show reproducible progress,explain the important adaptation issues,and get agreement on the next project direction. This is a progress-and-planning meeting;complete suite reproduction is not claimed.

## Slide1 — Goal and current scope
- Mentor roadmap:LIBERO -> ACT/DP/BAKU -> A2A.
- Current scope:oneLIBERO-Object task,pick soup and place it in the basket.
- Deliverable snapshot:working training/evaluation pipelines,pinned code/adapters,checkpoints/videos,and controlled diagnostics.
Speaker point:“I have a reproducible single-task pipeline,and I’m now separating implementation issues from robustness limitations.”

## Slide2 — What is implemented
-50demonstrations,40train/10validation,fixed20evaluationinitialstates,600stepcap.
- ACT:action chunks/CVAE;DP:action-sequence diffusion;BAKU:author transformerBC;A2A:flow from encoded proprioceptive history.
- Observation:twoRGBcameras+9Djoint/gripperfeedback.Action:7DOSCtranslation/rotation/grippercommand.
- Same causal labeling and data split;method-specific budgets,batches,execution and normalization remain explicit.
Speaker point:“These are author-code-based LIBERO adaptations;not identical recipes or equal-compute comparisons.”

## Slide3 — Important adaptation finding:temporal alignment
- LIBERO exporter savesobs[t]afteraction[t].
- Train next commands fromaction[t+1],not the alreadyexecutedaction[t].
- A2Amatchedseed0experiment:sameinit/sampleorder/budget,oldlabel0/20 ->correctlabel20/20.
- This establishes the effect in one matched setting;it does not establish multi-seedrobustness.
Visual:shorttimeline plus two explicit samples,obs[t]->action[t]andobs[t]->action[t+1].

## Slide4 — Current results:show both training seeds
| Recipe | Updates | Seed0 | Seed1 |
|---|---|---|---|
| ACT execute5 |10k|17/20|17/20|
| DP DDPM100 execute5(primary)|3k|15/20|12/20|
| DP DDPM100 execute1(prespecified secondary)|same weights|19/20|19/20|
| BAKU temporal aggregation|10k|20/20|6/20|
| A2A6flowsteps execute1|10k|20/20|0/20|

- Seed0A2Aadditionalchecks:1flowstep19/20;newinitialIDs40..49/6steps10/10.
- DPexecute1isreplanningafteronephysicalaction,still100denoisingsteps.ItisnotA2Aone-flow-stepinference.
- Single task,20rollouts/model,only2trainingseeds;no inference-latency reproduction and no broad superiority claim.

## Slide5 — A2A instability:what we have isolated
- Frozen checkpoint swap acrossV100/RTX6000:goodmodel20/20onboth,badmodel0/20onboth.InferenceGPUdoesnotexplainthisgap.
- Saved sources identical,normalizers identical,sampling coverage>99.7%,expert-windowAE/Flow errors close.
- Initial-stateinputisolation:onIDs20/38,swappingonlyexpertgripperfeedbackflipsseed1fromclose(+~.85)toopen(~-1);swappingarmjointsalonedoesnot.RawandEMAfirst-actiondirectionsshowthesamepattern.
- Recorded initialfingerfeedback:expert~(+.0362,-.0362),zero-warmup~(+.0205,-.0205). Most normalized initialstate distance herecomesfromgripperfeedback,notarmjoints.
- Physical2statepilot300stepcap:defaultseed0=2/2,seed1=0/2;explicitopen-gripperwarmupboth=2/2.
- IMPORTANT:openwarmupchangesinitialization. PinnedLIBEROreferenceusesall-zerodummyactions;originalbenchmarkresultsstayunchanged.Thisisadiagnosticintervention,notyetacanonicalfix.
- Completed40rolloutconfirmation14516670:bothfrozenmodels,sameV10032GB,20originalinitialstateseach,openwarmup,600stepcap,zero training. Seed0=20/20(mean129.5steps),seed1=20/20(mean128.2steps);all40videos/actions/config/weightsverified. Presentasdiagnosticchangedinitialization,alongsideoriginalzero-warmup20/20vs0/20.
Visual:one matchedfailure/successclip or contact sheet,then the smallpilot table.

## Slide6 — Discussion:agree on next project direction
1. Reproduction priority:howmuchstability/seed/taskcoverageisexpectedbeforemovingon?ForA2A,keepreferenceinitializationandimprovetrainingcoverage/representation,or explicitlystandardizeaninitializationprotocolforallmethods?
2. Research direction:DP'sfixedexec1vs5results motivate studying actiontiming,butadaptiveintervalsarenotyetimplemented/validated. Should we prioritize a small adaptive-timing experiment informed by the previously discussed Chronos/VFRTok ideas?
3. Scopeandmilestones:whichmethods/tasks/compute/reportformatshouldwefocusonnext?Clarify whetherBCinitializationfollowedbyRLrefinementistheintendedlaterdirection.

Speaker point:“I’d like to agree on one testable next hypothesis and a small evaluation plan,rather than expanding several directions simultaneously.”

## Morning checklist
- DONE:job14516670completed;40rollouts/config/weights/videoevidencevalidated,bothseeds20/20underdiagnosticopenwarmup.
- Preserveoriginalvsdiagnosticconditionsasseparatetables.
- Selectonegood/badmatchedvideoandonebriefcode/timelinediagram.
- Rehearseexplainingobs/actiontiming,exec-horizonvsflowsteps,andwhyofflineaveragelossdoesnotpredictclosed-loopsuccess.
- KeepjobIDs/checksums in backup material,not the main slide text.

Sources:
docs/causal-results-overview-2026-10-27.md
docs/a2a-seed-diagnosis-findings-2026-10-27.md
artifacts/a2a-initial-state-probe/14515931/result.json
docs/a2a-open-init-confirm-2026-10-27.md (created/updated when confirmation runs)
