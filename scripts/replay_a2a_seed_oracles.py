"""Matched CPU physics replay of GT, AE-oracle and expert-observation flow arrays."""
import argparse,hashlib,json,os,sys,time,traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument("--comparison-job",required=True);args=p.parse_args()
parent=ROOT/"artifacts/a2a-seed-comparison"/args.comparison_job
source=json.loads((parent/"result.json").read_text());assert source["completed"] and source["training_updates"]==0
path=parent/"comparison_arrays.npz"
h=hashlib.sha256()
with path.open("rb") as f:
 for block in iter(lambda:f.read(8*1024*1024),b""):h.update(block)
assert h.hexdigest()==source["arrays_sha256"]
arrays=np.load(path,allow_pickle=False)
job=os.environ["SLURM_JOB_ID"];out=ROOT/"artifacts/a2a-seed-oracle"/job;out.mkdir(parents=True,exist_ok=True)
r={"completed":False,"training_updates":0,"comparison_job":args.comparison_job,"arrays_sha256":h.hexdigest(),"cases":[]}
env=None;start=time.time()
def write(): (out/"result.json").write_text(json.dumps(r,indent=2)+"\n")
try:
 from libero.libero import benchmark,get_libero_path
 from libero.libero.envs.env_wrapper import ControlEnv
 suite=benchmark.get_benchmark_dict()["libero_object"]();task=suite.get_task(0)
 env=ControlEnv(bddl_file_name=str(Path(get_libero_path("bddl_files"))/task.problem_folder/task.bddl_file),
       use_camera_obs=False,has_renderer=False,has_offscreen_renderer=False)
 for demo in ("demo_28","demo_36"):
  gt=arrays["seed0_"+demo+"_gt"];np.testing.assert_array_equal(gt,arrays["seed1_"+demo+"_gt"])
  cases=[("GT",None,gt)]
  for seed in (0,1):
   for mode in ("ae","flow6"):cases.append((mode,seed,arrays["seed"+str(seed)+"_"+demo+"_"+mode]))
  gt_success=None
  for mode,seed,commands in cases:
   assert len(commands)==len(gt) and commands.shape[1:]==(7,)
   # A stored observation begins after command0; causal oracle keeps command0 GT.
   np.testing.assert_array_equal(commands[0],gt[0])
   env.seed(0);env.reset();env.set_init_state(arrays[demo+"_state0"])
   success=False;records=[]
   for step,action in enumerate(commands):
    _,reward,done,_=env.step(np.clip(action,-1,1))
    assert np.isfinite(env.sim.data.qpos).all()
    obs=env.env._get_observations();success=bool(env.check_success())
    records.append({"step":step,"action":action.tolist(),"reward":float(reward),"success":success,
       "eef_pos":obs["robot0_eef_pos"].tolist(),"gripper_qpos":obs["robot0_gripper_qpos"].tolist()})
    if success or done:break
   if mode=="GT":gt_success=success
   label=demo+"_"+mode+("_seed"+str(seed) if seed is not None else "")
   (out/(label+".json")).write_text(json.dumps(records)+"\n")
   row={"demo":demo,"mode":mode,"seed":seed,"success":success,"steps":len(records),
        "matching_GT_success":gt_success,"completed":True}
   r["cases"].append(row);write();print(json.dumps(row),flush=True)
 r["completed"]=True
 r["interpretation"]="AE uses future GT commands;flow uses prerecorded expert observations. Diagnostic oracles,not deployable policies or benchmark success. Interpret failure only when matched native GT replay succeeds."
except Exception as exc:
 r["error"]=repr(exc);traceback.print_exc()
finally:
 if env is not None:env.close()
 r["wall_seconds"]=time.time()-start;write();print(json.dumps(r,indent=2),flush=True)
sys.exit(0 if r["completed"] else 1)
