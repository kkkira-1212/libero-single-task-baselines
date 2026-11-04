# DP parallel experiments

### Parallel DP experiments submitted — execution interval and training seed
- User authorized several background jobs. Submitted four jobs, all explicit gts-agarg35/embers,1GPU/4CPU/24GB,knownECCnode excluded,emailBEGIN/END/FAIL.
-14353581: fixedseed0EMA execution_horizon1,20trialsIDs/seeds20..39,100DDPMsteps,600simulationstepcap,max6h.
-14353583: paired execution_horizon5 control,sameweights/trials/evaluator,max2h. Both interval jobs restricted gpu-rtx6000 for same nominal GPU type; do not interpret aggregate walltime as standalone inference latency.
-14353584: independentDP trainingseed1,3000updates,allotherconfigvalues identical toseed0,max2h.14353585:afterok dependent20trial evaluation,exec5,max2h.
- Frozen DPseed0 copy ~/scratch/robot_manipulation/frozen/dp_seed0_step3000/checkpoint.pt chmod0444; SHA25610c9b963fe52dbbd7e08b2e099811371cc1926b4c806ecbbce3a79d0f96424b1 verified before submission. Originalcheckpoint unchanged.
- Added --config/--checkpoint/--eval-protocol to experiments/dp_run.py. Training config is validated against checkpoint independently from narrowly allowed eval overrides; interval must be integer1..8, frozen checksum checked before model load,upstreamcommit checked. Report records actual execution interval separately from trainingconfig. Source snapshots saved with each newjob.
- Preflight AST,bashsyntax,seed-only config difference,and execution-interval/checksum wiring assertions passed. These are static checks; new override path awaits runtime verification.
- Interval1 vs5 changes feedback/replanning frequency and consumes different stochastic-noise sequences after the first prediction despite same trial seeds. It is a fixed-interval diagnostic,not an adaptive policy or proof of a timing mechanism. Eval IDs already used: report as development comparison.
- Keep all20paired outcomes,inspect placement behavior(ID38 particularly) without selecting only improved trials. Successrate may fall if frequent replanning causes inconsistent actions. Do not choose a winning interval and claim untouched confirmation.
- Expected logs/dp-eval-14353581.log,logs/dp-eval-14353583.log,logs/dp-train-14353584.log,logs/dp-eval-14353585.log; corresponding artifacts/dp-eval or dp-train/<job>/result.json and videos.
- User viewed full ID38 animation and clarified failure occurs at final basket placement; record this as placement-stage failure. Exact geometric/contact/release cause remains unmeasured.
