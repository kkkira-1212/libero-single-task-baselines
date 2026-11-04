"""Frozen A2A inference-hardware cross-check; training hardware/seed remain confounded."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(path):
 p=ROOT/path
 return json.loads(p.read_text()) if p.exists() else {}
manifest=read("configs/a2a/hardware_check_2026-10-27.json")
report={"scope":manifest["purpose"],"cases":{}}
for name,case in manifest["cases"].items():
 frozen=read(case["frozen_manifest"])
 ev=read("artifacts/a2a-eval/"+case["job_id"]+"/result.json")
 ref=read("artifacts/a2a-eval/"+case["reference_evaluation_job"]+"/result.json")
 row=dict(case,completed=bool(ev.get("completed")),successes=None,success_rate=None,
          reference_successes=ref["successes"],reference_gpu=ref["gpu"])
 if row["completed"]:
  assert ev["config"]==frozen["config"] and ev["checkpoint_sha256"]==frozen["checkpoint_sha256"]
  assert ev["sampling_steps"]==6 and ev["execution_horizon"]==1
  assert [r["initial_state_id"] for r in ev["rollouts"]]==list(range(20,40))
  assert [r["seed"] for r in ev["rollouts"]]==list(range(20,40))
  folder=ROOT/"artifacts/a2a-eval"/case["job_id"]
  for r in ev["rollouts"]:
   assert r["completed"] and 0<r["steps"]<=600
   with (folder/("trial_%s.gif"%r["initial_state_id"])).open("rb") as f:assert f.read(6) in (b"GIF87a",b"GIF89a")
   actions=read(folder/("trial_%s_actions.json"%r["initial_state_id"]))
   assert len(actions)==r["steps"] and all(len(a)==7 for a in actions)
  successes=sum(r["success"] for r in ev["rollouts"]);assert successes==ev["successes"]
  matches=case["expected_gpu_token"] in ev["gpu"].upper().replace(" ","")
  row.update(successes=successes,success_rate=successes/20,gpu=ev["gpu"],checkpoint_sha256=ev["checkpoint_sha256"],
             evaluation_completed=True,hardware_matches_protocol=matches,completed=matches)
 report["cases"][name]=row
report["completed"]=all(r["completed"] for r in report["cases"].values())
out=ROOT/"artifacts/a2a-hardware-check/2026-10-27";out.mkdir(parents=True,exist_ok=True)
(out/"result.json").write_text(json.dumps(report,indent=2)+"\n")
lines=["# Frozen A2A inference-hardware cross-check","",
 "Seed0 train/eval RTX6000:20/20. Seed1 train/eval V100:0/20.",
 "These jobs move each fixed checkpoint to the other inference GPU type,with unchanged6flowsteps/exec1/IDs20..39/seeds20..39.","",
 "| Case | Job | Reference | Cross-hardware result |","|---|---|---|---|"]
for name,row in report["cases"].items():
 score=str(row["successes"])+"/20" if row["completed"] else "pending/incomplete"
 lines.append("| "+name+" | "+row["job_id"]+" | "+str(row["reference_successes"])+"/20 | "+score+" |")
lines+=["","Previous attempt14498426 executed on V100 despite the requested RTX6000 partition; retained as a V100 repeat, not the intended RTX6000 check.","","No training updates. This checks inference hardware effects; it does not isolate training hardware from training seed."]
text="\n".join(lines)+"\n";(out/"result.md").write_text(text)
(ROOT/"docs/a2a-hardware-check-2026-10-27.md").write_text(text)
print(json.dumps(report,indent=2))
