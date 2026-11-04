"""Diagnostic open-gripper initialization confirmation; original benchmark unchanged."""
import hashlib,json,os,random,sys,time,traceback
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.a2a_confirm_eval import evaluate_confirm
job=os.environ["SLURM_JOB_ID"];out=ROOT/"artifacts/a2a-open-init-confirm"/job;out.mkdir(parents=True,exist_ok=True)
r={"completed":False,"training_updates":0,"scope":"Diagnostic altered initialization;do not replace original zero-warmup primary results",
   "initial_state_ids":list(range(20,40)),"rollouts_per_seed":20,"settle_steps":10,"settle_gripper_command":-1.0,
   "rollout_horizon":600,"sampling_steps":6,"execution_horizon":1,"seeds":{}}
start=time.time()
def write():
 (out/"result.json").write_text(json.dumps(r,indent=2)+"\n")
 lines=["# A2A open-gripper initialization confirmation","",
   "Changed diagnostic initialization,not a replacement for the original all-zero warmup scores.",
   "Same frozen10kEMAweights,IDs20..39/seeds20..39,6flowsteps,execute1,settle10,horizon600. No training.",
   "Both checkpoints are evaluated on the same allocatedGPU.","",
   "| Training seed | Success | Job / artifact |","|---|---|---|"]
 for seed,item in r["seeds"].items():
  score=str(item["successes"])+"/20" if item["completed"] else "incomplete"
  lines.append("| "+seed+" | "+score+" | "+job+"/seed"+seed+" |")
 lines+=["","Original zero-warmup results remain seed0=20/20,seed1=0/20.",
   "Compare the full conditions and report remaining training-GPU/seed confounding and single-task scope."]
 (ROOT/"docs/a2a-open-init-confirm-2026-10-27.md").write_text("\n".join(lines)+"\n")
try:
 assert torch.cuda.is_available();torch.set_num_threads(4)
 r["gpu"]=torch.cuda.get_device_name(0)
 for source in ("experiments/a2a_open_init_confirm.py","experiments/a2a_confirm_eval.py"):
  (out/Path(source).name).write_text((ROOT/source).read_text())
 for seed in (0,1):
  frozen=json.loads((ROOT/("configs/a2a/frozen_causal_seed"+str(seed)+"_10000.json")).read_text())
  dest=out/("seed"+str(seed));dest.mkdir(exist_ok=True)
  protocol={"purpose":r["scope"],"sampling_steps":6,"execution_horizon":1,"settle_gripper_command":-1.0,
   "checkpoint_sha256":frozen["checkpoint_sha256"]}
  random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
  evaluate_confirm(frozen["config"],dest,Path(frozen["checkpoint"]),job,protocol)
  e=json.loads((dest/"result.json").read_text())
  assert e["completed"] and e["completed_rollouts"]==20 and e["checkpoint_sha256"]==frozen["checkpoint_sha256"]
  assert e["gpu"]==r["gpu"] and [x["initial_state_id"] for x in e["rollouts"]]==list(range(20,40))
  for row in e["rollouts"]:
   assert row["completed"]
   with (dest/("trial_%s.gif"%row["initial_state_id"])).open("rb") as f:assert f.read(6) in (b"GIF87a",b"GIF89a")
   actions=json.loads((dest/("trial_%s_actions.json"%row["initial_state_id"])).read_text())
   assert len(actions)==row["steps"] and all(len(a)==7 for a in actions)
  r["seeds"][str(seed)]={"completed":True,"successes":e["successes"],"rollouts":20,
   "checkpoint_sha256":e["checkpoint_sha256"],"report":str(dest/"result.json")}
  write();print(json.dumps(r["seeds"][str(seed)]),flush=True)
 r["completed"]=True
except Exception as exc:
 r["error"]=repr(exc);traceback.print_exc()
finally:
 r["wall_seconds"]=time.time()-start;write();print(json.dumps(r,indent=2),flush=True)
sys.exit(0 if r["completed"] else 1)
