# ACT / DP / BAKU causal-v1 baseline experiment plan

Date:2026-10-27. User authorized preparing and queueing corrected-timing baseline experiments in the background.

## Contract and implementation
LIBERO obs[t] is saved after command[t]. The next executable prediction must start at command[t+1].

- ACT: retain current9Dstate/twoimages;chunk20starts at action[t+1]. Preserve valid-action mask and normalized zero padding;exclude final observation. Validation uses eligible-anchor lengths.
- DP: retain two observations[t-1,t];shift packed16actions to[t,...,t+15]. The unchanged author output slice[1:9] then predicts[t+1,...,t+8]. Preserve boundary repetition.
- BAKU: retain the original PKLs/embedding/official episode-uniform sampler. It already excludes the final observation. Local CausalBCDataset shifts each loaded episode's actions left by1 and appends a zero command. Query10 now starts at[t+1];preserve author zero tail padding/nearidentity scaling/temporal aggregation. No official source or PKL changes.

All use the same40train/10validation demos and6156eligible training anchors. BAKU episode-uniform sampling remains different from ACT/DP timestep-uniform sampling. Each method keeps its train-only normalization and image preprocessing. Legacy configurations preserve old defaults; previous modified sources are archived under artifacts/baselines-causal-change/source-before/.

## Prespecified recipes
| Method | Seeds | Updates per seed | Batch | Main execution | Checkpoint |
|---|---|---|---|---|---|
| ACT | 0,1 | 10000 | 16 | execute5 from chunk20 | final10k,not validation-best |
| DP | 0,1 | 3000 | 16 | execute5,DDPM100 | final3kEMA |
| BAKU | 0,1 | 10000 | 16 | every-step temporal aggregation,query10 | final10k |

Each evaluation uses same task0,IDs20..39,seeds20..39,settle10,600stepcap,20rollouts. DP also has a prespecified secondary execute1 evaluation of each same final checkpoint.

Common causal labeling/data/evaluation scope does not mean equal compute or identical method recipes. A2A uses history8/batch4/10k/exec1;report these differences explicitly. Historical results are not a paired timing-only ablation for these three methods. No suite-level reproduction claim.

## Gates/resources
Actual-HDF5 gates check200representative start/middle/end windows per method,correct future labels,unchanged observations and legacy compatibility. GPU gates also check finite update,prediction and strict save/reload;ACT's actual action encoder and DP's actual diffusion target are inspected. BAKU PKL hashes remain unchanged.

Each train waitsafterokitsownGPUgate. Runtime verifies tested source SHA and snapshots relevant code/environment setup. Failure blocks dependent training. Old checkpoints/results are retained.

Only gts-agarg35/embers;known ECC nodes excluded;V100/RTX6000/A100,notL40S. CPUchecks2CPU16GB20min;GPUgates1GPU4CPU24GB30min;train1GPU4CPU24GB2h;eval1GPU4CPU24GB3h. BEGIN/END/FAILemailxinwang915@gatech.edu. No installs/downloads.

## Jobs
| Method | CPU data | GPU gate | Seed0 train->eval | Seed1 train->eval |
|---|---|---|---|---|
| ACT | 14497115 | 14497117 | 14497118->14497119 | 14497120->14497121 |
| DP | 14497122 | 14497123 | 14497124->14497125 | 14497127->14497129 |
| BAKU | 14497136 | 14497137 | 14497138->14497139 | 14497140->14497141 |

DPsecondaryexec1evals14497126(seed0),14497132(seed1).
Collector14497224afterany8evals;partialjobsarenotscoredasfailures.

Manifest:configs/baselines_causal/manifest_2026-10-27.json.
Results:docs/baselines-causal-2026-10-27.md.
Refresh:python3 scripts/summarize_causal_baselines.py.

## Verified milestone
AllthreeCPUchecks passed200windows each,total600,zero trainingupdates. AllthreeGPUgatespassedoneCUDAupdate/prediction/reload(maxerror0.0);all6trainjobsstarted. Dependentevalsnotyetcompleted. Gate passes are interface evidence,not learned-policy success.

ThursdayOct29exam/SundayNov1summary/MondayNov2reporttargetretained. No budget/architecture/physical-prior search.
