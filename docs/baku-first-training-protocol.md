# First BAKU single-task training protocol

### First BAKU training and dependent evaluation submitted
- User authorized prepare/continueexperiments afterdata/model/livegatespassed.
- Fixed configs/baku/train_seed0.json:seed0,3000updates,batch16,constantAdamWLR1e-4/weightdecay1e-4 via official BCAgent optimizers;no newgradientclipping/scheduler/algorithm modifications. Fresh initialization,not resume1-update diagnosticcheckpoint.
- Model exactlymatches verifiedmodelconfig:sharedrandom-initResNet/FiLM,GPT8layers hidden256,language384,proprio9,action7,historyfalse,10querychunks,deterministichead,std.1,officialtemporalaggregation eachsimulationstep.
- Official BCDataset train40/validation10,episodeuniform+validwindow sampling,zeroactionpadding,nearidentitystats andflippedimages retained. UnlikeDPtimestepuniformsampling/limitsstats;not an architecture-only controlled comparison.
- Source datasetandbothPKLhasheschecked atjobstartup;originaltask/split preserved. Model interface/config sourcedfrompassed14354835;livegate14373868 passed.
- experiments/baku_run.py calls authorupdate/act;fixed32validation samples collected under restoredRNG;validation at0/every500;checkpoint every500 withmodelstate andoptimizerstate_dicts separately plusPython/NumPy/Torch/CUDARNG. Exactresumerunnernotimplemented;retainedstates supportfuturecontinuation.
- Finalpredetermined3000-updatecheckpoint selected,no loss/online-basedselection. EvalIDs/seeds20..39,600stepcap,10settlezeros,perstepofficialtemporalaggregation,actionclip[-1,1],20GIFs/actionfiles,partialreports/finalWilson95%. These areuseddevelopmentIDs,notuntouchedconfirmation.
- Submitted train14386819 andafterok eval14386820;each1GPU/4CPU/24GB/max2h,explicitgts-agarg35/embers,knownECCnodeexcluded,emailBEGIN/END/FAIL. CPU/PythonAST/bashsyntax/configgatespassed;fullnewtraining/evalrunnerawaitsruntimeverification.
- Checkpoint scratch/robot_manipulation/baku-training/14386819/latest.pt. Evidence logs/baku-train-14386819.log,artifacts/baku-train/14386819/metrics.json,result.json;logs/baku-eval-14386820.log,artifacts/baku-eval/14386820/result.json/GIFs.
- Scope isfirstsingle-taskpilot,not BAKUpaper/suite reproduction. No A2Aexperimentsstarted.

### BAKU 10k-update controlled budget run submitted
- User authorized10k-updateexperiment. configs/baku/train_seed0_10000.json differsfromoriginalonlyinupdates3000->10000anddescriptiveselectionmetadata3000->10000;model/optimizer/sampling/batch/seed/data/evaluationunchanged.
- Freshseed0initializationrerunusingexistingverifiedrunner,notcheckpointcontinuation. Added--configCLIwithoriginaldefaultpreserved. Original3kcheckpoint/config/0of20resultskept.
- Bothnewjobsrestrictedgpu-rtx6000tomatchoriginal3kGPUtype;exactbitwise3kprefixagreementnotassumed acrossmachines.
- Finaltrain14390364,afterok20trialeval14390365;1GPU/4CPU/24GB,max2h each,freegts-agarg35/embers,knownECCnodeexcluded,mailBEGIN/END/FAIL.
- Initialsubmissions14390358/14390359werereplacedpromptlyafterdetectingcopiedselectiondescriptionstillread3000updates;cancelledbothbeforechangingconfigtoavoidrunningconfigrace. Numericalupdateguardalready10000;replacementensuressavedmetadataalsoaccurate. Cancellationemailsareexpected,notnewtrainingfailure.
- Validateevery500/savecheckpoints;exact10000-stepfinalmodelselectedbeforeevaluationIDs/seeds20..39,600steps/10settle,officialtemporalaggregation. Primaryquestioniswhetheradditionaltrainingimprovesthefailedpilot,notconfirmationofundertrainingbeforeoutcome.
- AST/bashsyntaxandconfigdiffcheckspassed;runtimepending. Logs/baku-train-14390364.log,logs/baku-eval-14390365.log;artifacts/baku-train/14390364/result.json,artifacts/baku-eval/14390365/result.json.
