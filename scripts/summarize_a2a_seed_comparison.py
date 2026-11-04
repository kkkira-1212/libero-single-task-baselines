"""Read-only summary of paired frozen A2A seed diagnostics and oracle replays."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def read(path):
 p=ROOT/path
 return json.loads(p.read_text()) if p.exists() else {}
m=read("configs/a2a/seed_comparison_2026-10-27.json")
full=read("artifacts/a2a-seed-comparison/"+m["offline_job"]+"/result.json")
small=read("artifacts/a2a-seed-comparison/"+m["cpu_preliminary_job"]+"/result.json")
oracle=read("artifacts/a2a-seed-oracle/"+m["oracle_job"]+"/result.json")
static=read("artifacts/a2a-seed-comparison/static/source_and_sampling.json")
main=full if full.get("completed") else small
mainjob=m["offline_job"] if full.get("completed") else m["cpu_preliminary_job"]
r={"training_updates":0,"full_comparison_completed":bool(full.get("completed")),
   "cpu_preliminary_completed":bool(small.get("completed")),"oracle_completed":bool(oracle.get("completed")),
   "errors":{name:v.get("error") for name,v in (("full",full),("cpu_preliminary",small),("oracle",oracle)) if v.get("error")},
   "source_and_sampling":static,"phase_comparison":{}}
if main.get("completed"):
 arrays=np.load(ROOT/"artifacts/a2a-seed-comparison"/mainjob/"comparison_arrays.npz",allow_pickle=False)
 for subset,meta in main["sample_plan"].items():
  np.testing.assert_array_equal(arrays["0_ema_policy_"+subset+"_target"],arrays["1_ema_policy_"+subset+"_target"])
  groups={"first_repeated_history":[x["t"]<8 for x in meta],
          "gripper_switch":[x["gripper_switch"] for x in meta],
          "all":[True]*len(meta)}
  r["phase_comparison"][subset]={}
  for phase,mask in groups.items():
   mask=np.array(mask,dtype=bool);row={"samples":int(mask.sum()),"seeds":{}}
   for seed in ("0","1"):
    targets=arrays[seed+"_ema_policy_"+subset+"_target"][mask]
    results={}
    for mode in ("ae","flow1","flow6"):
     pred=arrays[seed+"_ema_policy_"+subset+"_"+mode][mask]
     results[mode]={"first_translation_rmse":float(np.sqrt(np.mean((pred[:,0,:3]-targets[:,0,:3])**2))) if mask.any() else None,
       "first_gripper_sign_accuracy":float(np.mean(np.sign(pred[:,0,6])==np.sign(targets[:,0,6]))) if mask.any() else None}
    row["seeds"][seed]=results
   r["phase_comparison"][subset][phase]=row
 r["normalizer_max_abs_difference_vs_seed0_EMA"]=main["normalizer_max_abs_difference_vs_seed0_EMA"]
r["oracle_cases"]=oracle.get("cases",[])
r["same_live_initial_observation_probe"]=full.get("same_live_initial_observation_probe")
r["live_probe_error"]=full.get("live_probe_error")
out=ROOT/"artifacts/a2a-seed-comparison/summary";out.mkdir(parents=True,exist_ok=True)
(out/"result.json").write_text(json.dumps(r,indent=2)+"\n")
lines=["# Frozen A2A seed diagnosis results","",
 "No training updates. Same selected expert inputs,unchanged checkpoints. Training GPU and seed were not independently controlled.","",
 "Full raw/EMA comparison complete:"+str(r["full_comparison_completed"])+". CPU preliminary complete:"+str(r["cpu_preliminary_completed"])+". Oracle complete:"+str(r["oracle_completed"])+".",
 "CPU preliminary covers32train/16validation windows only;fullcomparison has broader prespecified phase coverage.","",
 "Actual training source snapshots are identical;both40000drawsequences were reconstructed with exact SHA matches.",
 "Seed0/1coverage99.756%/99.838%;gripper-switchdraws538/530. Seed1sawall86switchanchors.","",
 "| Weight | Split | Seed | Windows | AE first translation RMSE | Flow6 first translation RMSE | Flow6 first grip-sign accuracy |",
 "|---|---|---|---|---|---|---|"]
for seed,model in main.get("models",{}).items():
 for weight,subsets in model.get("weights",{}).items():
  for subset,v in subsets.items():
   a=v["ae"];f=v["flow6"]
   lines.append("| "+" | ".join(map(str,[weight,subset,seed,a["samples"],round(a["first_translation_rmse"],6),round(f["first_translation_rmse"],6),round(f["first_gripper_sign_accuracy"],4)]))+" |")
lines+=["","Errors above are in raw controller-command input units,not measured robot-position error.",
 "Latent coordinate bases can differ across trained models;compare decoded actions rather than swapping cross-seed encoders/decoders.","",
 "| Demonstration | Mode | Seed | Success | Steps | Matched GT succeeds |",
 "|---|---|---|---|---|---|"]
for c in r["oracle_cases"]:
 lines.append("| "+" | ".join(map(str,[c["demo"],c["mode"],c["seed"],c["success"],c["steps"],c["matching_GT_success"]]))+" |")
lines+=["","AE oracle uses future ground-truth actions;flow oracle uses prerecorded expert observations. These are diagnostic teacher-forced replays,not deployable-policy success or benchmark results.",
 "Two demonstration replays do not establish all-task/component correctness. No unique failure cause is automatically inferred.",
 "Closed-loop contact sheet:artifacts/a2a-seed-comparison/static/closed_loop_frames.png.",
 "Machine-readable phase metrics:artifacts/a2a-seed-comparison/summary/result.json."]
text="\n".join(lines)+"\n"
(out/"result.md").write_text(text);(ROOT/"docs/a2a-seed-comparison-results-2026-10-27.md").write_text(text)
print(json.dumps({k:v for k,v in r.items() if k not in ("source_and_sampling","normalizer_max_abs_difference_vs_seed0_EMA")},indent=2))
