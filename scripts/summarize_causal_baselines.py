"""Evidence-backed causal baseline report; partial jobs have no scored success rate."""
import hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(path):
 p=ROOT/path
 return json.loads(p.read_text()) if p.exists() else {}
manifest=read("configs/baselines_causal/manifest_2026-10-27.json")
report={"scope":"common task/split/causal timing/evaluation IDs; method-specific budgets and recipes",
        "runs":[],"all_primary_completed":True}
hash_cache={}
def checkpoint_digest(path):
 if path not in hash_cache:
  h=hashlib.sha256()
  with Path(path).open("rb") as f:
   for block in iter(lambda:f.read(8*1024*1024),b""):h.update(block)
  hash_cache[path]=h.hexdigest()
 return hash_cache[path]
def inspect(method,seed,entry,job,execution,secondary=False):
 c=read(entry["config"]);prefix="act-pilot" if method=="act" else method+"-train"
 training=read("artifacts/"+prefix+"/"+entry["training_job"]+"/result.json")
 ev=read("artifacts/"+method+"-eval/"+job+"/result.json")
 complete=bool(ev.get("execution_ok" if method=="act" else "completed"))
 trained=bool(training.get("completed_target" if method=="act" else "completed"))
 row={"method":method,"seed":seed,"training_job":entry["training_job"],"evaluation_job":job,
      "updates":c["updates"],"batch_size":c["batch_size"],"secondary":secondary,"execution":execution,
      "training_completed":trained,"evaluation_completed":complete,
      "recorded_rollouts":len(ev.get("rollouts",[])),"successes":None,"success_rate":None}
 if complete:
  assert trained and training["config"]==c and ev["config"]==c
  assert training["checkpoint"]==ev["checkpoint"]
  rows=ev["rollouts"];assert len(rows)==20 and all(r["completed"] for r in rows)
  assert [r["initial_state_id"] for r in rows]==list(range(20,40))
  assert [r["seed"] for r in rows]==list(range(20,40))
  assert ev["checkpoint_training_step"]==c["updates"]
  folder=ROOT/"artifacts"/(method+"-eval")/job
  protocol=read(folder/"causal_protocol.json")
  assert protocol["sample_contract"]=="libero-poststep-next-command-v1"
  gate=read("artifacts/baseline-causal-check/"+protocol["gate_job"]+"/result.json")
  assert gate["passed"] and gate["method"]==method and gate["training_updates"]==1
  assert protocol["source_sha256"]==gate["source_sha256"]
  for source,sha in protocol["source_sha256"].items():
   assert hashlib.sha256((folder/("source_"+source.replace("/","__"))).read_bytes()).hexdigest()==sha
  for r in rows:
   i=r["initial_state_id"]
   stem="rollout_" if method=="act" else "trial_"
   with (folder/(stem+str(i)+".gif")).open("rb") as f:assert f.read(6) in (b"GIF87a",b"GIF89a")
   path=folder/("rollout_"+str(i)+".json" if method=="act" else "trial_"+str(i)+"_actions.json")
   actions=read(path)
   assert len(actions)==r["steps"] and 0<r["steps"]<=600
   for a in actions:
    a=a["action"] if isinstance(a,dict) else a
    assert len(a)==7 and all(math.isfinite(v) and -1<=v<=1 for v in a)
  successes=sum(r["success"] for r in rows)
  assert ev["successes"]==successes and ev["success_rate"]==successes/20
  if method=="dp":assert ev["execution_horizon"]==int(execution)
  row.update(successes=successes,success_rate=successes/20,wilson95=ev["wilson95"],
    checkpoint=ev["checkpoint"],
    checkpoint_sha256=ev.get("checkpoint_sha256") or checkpoint_digest(ev["checkpoint"]),
    checksum_scope="post-evaluation file verification" if method=="act" else "verified in evaluation")
 return row
for method,group in manifest["methods"].items():
 for seed,entry in group["runs"].items():
  execution="temporal aggregation" if method=="baku" else "5"
  row=inspect(method,int(seed),entry,entry["primary_evaluation_job"],execution)
  report["runs"].append(row)
  report["all_primary_completed"] &= row["evaluation_completed"]
  if "secondary_exec1_evaluation_job" in entry:
   report["runs"].append(inspect(method,int(seed),entry,entry["secondary_exec1_evaluation_job"],"1",True))
report["all_evaluations_completed"]=all(r["evaluation_completed"] for r in report["runs"])
out=ROOT/"artifacts/baselines-causal/2026-10-27";out.mkdir(parents=True,exist_ok=True)
(out/"result.json").write_text(json.dumps(report,indent=2)+"\n")
lines=["# Causal-v1 ACT, Diffusion Policy and BAKU experiments","",
 "Same task,40train/10validation demos,obs[t] -> next command[t+1],IDs20..39,settle10,600stepcap.",
 "Each method retains its prior budget/model/optimizer/normalizer/padding/sampling/execution recipe.",
 "This is not equal-compute or original suite-level reproduction. Historical numbers are not a paired timing ablation.","",
 "| Method | Training seed | Updates | Batch | Execution | Training job | Eval job | Result |",
 "|---|---|---|---|---|---|---|---|"]
for row in report["runs"]:
 if row["secondary"]:continue
 score=str(row["successes"])+"/20" if row["evaluation_completed"] else "pending/incomplete"
 lines.append("| "+" | ".join(map(str,[row["method"],row["seed"],row["updates"],row["batch_size"],row["execution"],row["training_job"],row["evaluation_job"],score]))+" |")
lines+=["","DP exec1 is a prespecified secondary frozen-weight evaluation, not a best-result selection.",""]
for row in report["runs"]:
 if row["secondary"]:
  score=str(row["successes"])+"/20" if row["evaluation_completed"] else "pending/incomplete"
  lines.append("- DP seed"+str(row["seed"])+", exec1, eval"+row["evaluation_job"]+": "+score)
lines+=["","Incomplete trials are not scored as failures. Failed gate jobs cannot release dependent training.",
 "ACT uses final10k checkpoint, DP final3kEMA, BAKU final10k; both training seeds must be reported.",
 "A2A causal seed0 reference20/20 at6flowsteps and19/20 at1step;training batch4/10k/exec1 differs from these recipes. Its seed1 confirmation remains in docs/a2a-confirmation-2026-10-27.md."]
text="\n".join(lines)+"\n";(out/"result.md").write_text(text)
(ROOT/"docs/baselines-causal-2026-10-27.md").write_text(text)
print(json.dumps(report,indent=2))
