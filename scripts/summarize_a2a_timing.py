"""Summarize the prespecified paired A2A timing experiment without scoring partial jobs."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/"configs/a2a/timing_comparison_2026-10-26.json").read_text())
def read(path):
 p=ROOT/path
 return json.loads(p.read_text()) if p.exists() else {}
gate=read("artifacts/a2a-causal-integration/"+manifest["gate_job"]+"/result.json")
report={"scope":"single-task, seed0, timing-only diagnostic; not paper reproduction",
        "gate_job":manifest["gate_job"],"gate_passed":gate.get("passed"),
        "gate_status":"passed" if gate.get("passed") else ("failed" if gate.get("error") else "pending"),"variants":{}}
data=read("artifacts/a2a-causal-data/"+manifest.get("data_check_job","unknown")+"/result.json")
report["data_check_passed"]=data.get("passed")
trains={};configs={}
for name,jobs in manifest["variants"].items():
 c=read(jobs["config"]);configs[name]=c
 t=read("artifacts/a2a-train/"+jobs["train_job"]+"/result.json");trains[name]=t
 e=read("artifacts/a2a-eval/"+jobs["eval_job"]+"/result.json")
 rows=e.get("rollouts",[])
 complete=bool(e.get("completed"))
 row=dict(jobs,training_completed=bool(t.get("completed")),evaluation_completed=complete,
          observed_completed_rollouts=len(rows),successes=None,success_rate=None,
          sample_order_sha256=t.get("sample_order_sha256"),checkpoint_sha256=e.get("checkpoint_sha256"))
 if complete:
  assert t.get("completed") and t["config"]==c and e["config"]==c
  assert t["checkpoint"]==e["checkpoint"]
  assert [r["initial_state_id"] for r in rows]==c["rollout_initial_state_ids"]
  assert len(rows)==20 and all(r.get("completed") for r in rows)
  assert e["sampling_steps"]==6 and e["execution_horizon"]==1
  row.update(successes=sum(r["success"] for r in rows),success_rate=e["success_rate"])
 report["variants"][name]=row
canonical=[{k:v for k,v in c.items() if k not in ("sample_contract","purpose")} for c in configs.values()]
paired=all(t.get("completed") for t in trains.values()) and len(canonical)==2 and canonical[0]==canonical[1]
hashes=[t.get("sample_order_sha256") for t in trains.values()]
paired=bool(paired and hashes[0] and len(set(hashes))==1 and all(t.get("training_samples")==6156 for t in trains.values()))
report["paired_training_conditions_verified"]=paired
report["completed"]=all(v["evaluation_completed"] for v in report["variants"].values())
report["success_rate_delta_causal_minus_control"]=(
 report["variants"]["causal"]["success_rate"]-report["variants"]["control"]["success_rate"]
 if report["completed"] and paired else None)
out=ROOT/"artifacts/a2a-timing-comparison/2026-10-26";out.mkdir(parents=True,exist_ok=True)
(out/"result.json").write_text(json.dumps(report,indent=2)+"\n")
lines=["# A2A timing-only paired experiment","",
       "Single LIBERO task, seed0, 10000 updates, 6 flow steps, execute1; fixed IDs20..39.",
       "Only target-label timing differs; both variants exclude the same terminal observation anchors.",
       "No paper reproduction, multi-seed robustness or causal physical-prior improvement is claimed.","",
       "Data check passed: "+str(report["data_check_passed"])+". GPU gate status: "+report["gate_status"]+". Paired actual sample order verified: "+str(paired)+".","",
       "| Variant | Training job | Evaluation job | Completed rollouts | Success |",
       "|---|---|---|---|---|"]
for name,row in report["variants"].items():
 score=str(row["successes"])+"/20" if row["evaluation_completed"] else "pending/incomplete"
 lines.append("| "+name+" | "+row["train_job"]+" | "+row["eval_job"]+" | "+str(row["observed_completed_rollouts"])+" | "+score+" |")
lines+=["","Partial runs are not scored as zero. A numerical paired comparison requires both complete evaluations and matching actual training sample-order digests.","",
        "Manifest: configs/a2a/timing_comparison_2026-10-26.json; raw reports remain under artifacts/a2a-{train,eval}/<job>."]
text="\n".join(lines)+"\n"
(out/"result.md").write_text(text)
(ROOT/"docs/a2a-timing-comparison-2026-10-26.md").write_text(text)
print(json.dumps(report,indent=2))
