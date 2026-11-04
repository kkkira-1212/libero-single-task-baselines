"""Summarize prespecified A2A confirmation jobs; partial evaluations are never scored."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(path):
 p=ROOT/path
 return json.loads(p.read_text()) if p.exists() else {}
manifest=read("configs/a2a/confirmation_2026-10-27.json")
frozen=read(manifest["frozen_manifest"])
training=read("artifacts/a2a-train/"+manifest["seed1_training"]["job_id"]+"/result.json")
report={"scope":manifest["scope"],"baseline":{"job_id":manifest["baseline_eval_job"],
 "successes":20,"rollouts":20,"sampling_steps":6,"training_seed":0},
 "seed1_training_completed":bool(training.get("completed")),"evaluations":{}}
for name,entry in manifest["evaluations"].items():
 protocol=read(entry["protocol"]);ev=read("artifacts/a2a-eval/"+entry["job_id"]+"/result.json")
 config=frozen["config"] if entry["frozen"] else read(manifest["seed1_training"]["config"])
 ids=protocol.get("initial_state_ids",config["rollout_initial_state_ids"])
 seeds=protocol.get("rollout_seeds",config["rollout_seeds"])
 row={"job_id":entry["job_id"],"completed":bool(ev.get("completed")),"rollouts_expected":len(ids),
      "rollouts_recorded":len(ev.get("rollouts",[])),"successes":None,"success_rate":None,
      "sampling_steps":protocol["sampling_steps"],"training_seed":config["seed"],
      "initial_state_ids":ids}
 if row["completed"]:
  assert ev["config"]==config and ev["rollout_initial_state_ids"]==ids and ev["rollout_seeds"]==seeds
  assert ev["sampling_steps"]==protocol["sampling_steps"] and ev["execution_horizon"]==1
  assert len(ev["rollouts"])==len(ids) and [r["initial_state_id"] for r in ev["rollouts"]]==ids
  assert [r["seed"] for r in ev["rollouts"]]==seeds
  if entry["frozen"]:
   assert ev["checkpoint_sha256"]==frozen["checkpoint_sha256"]
  else:
   assert training["completed"] and training["config"]==config and training["checkpoint"]==ev["checkpoint"]
  folder=ROOT/"artifacts/a2a-eval"/entry["job_id"]
  for source,sha in manifest["source_sha256"].items():
   assert hashlib.sha256((folder/Path(source).name).read_bytes()).hexdigest()==sha,source
  for r in ev["rollouts"]:
   assert r["completed"] and 0<r["steps"]<=config["rollout_horizon"]
   with (folder/("trial_%s.gif"%r["initial_state_id"])).open("rb") as f:assert f.read(6) in (b"GIF87a",b"GIF89a")
   actions=read(folder/("trial_%s_actions.json"%r["initial_state_id"]))
   assert len(actions)==r["steps"] and all(len(a)==7 for a in actions)
  s=sum(r["success"] for r in ev["rollouts"])
  assert s==ev["successes"] and ev["success_rate"]==s/len(ids)
  row.update(successes=s,success_rate=s/len(ids),checkpoint_sha256=ev["checkpoint_sha256"],
             wilson95=ev["wilson95"])
 report["evaluations"][name]=row
report["completed"]=all(row["completed"] for row in report["evaluations"].values())
out=ROOT/"artifacts/a2a-confirmation/2026-10-27";out.mkdir(parents=True,exist_ok=True)
(out/"result.json").write_text(json.dumps(report,indent=2)+"\n")
lines=["# A2A causal-v1 confirmation experiments","",
 "Reference: frozen seed0 model, 6 flow steps, IDs20..39, 20/20 successes.",
 "All evaluations use execution horizon1 and600stepcap. Different questions remain separate; no pooling.","",
 "| Experiment | Job | Flow steps | Expected rollouts | Result |",
 "|---|---|---|---|---|"]
for name,row in report["evaluations"].items():
 result=str(row["successes"])+"/"+str(row["rollouts_expected"]) if row["completed"] else "pending/incomplete"
 lines.append("| "+name+" | "+row["job_id"]+" | "+str(row["sampling_steps"])+" | "+str(row["rollouts_expected"])+" | "+result+" |")
lines+=["","No latency is measured in these experiments. New initial states are within the same task.",
 "Existing ACT/DP/BAKU results use the historical data convention; a corrected-protocol cross-method ranking requires a consistent timing audit/adaptation.",
 "Plan: exam Thursday Oct29; aim to consolidate Sunday Nov1 and report Monday Nov2."]
text="\n".join(lines)+"\n"
(out/"result.md").write_text(text)
(ROOT/"docs/a2a-confirmation-2026-10-27.md").write_text(text)
print(json.dumps(report,indent=2))
