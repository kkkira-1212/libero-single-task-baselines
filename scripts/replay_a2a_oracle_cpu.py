"""CPU-only oracle action replay: no rendering, no model inference, no parameter updates."""
import hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/"artifacts/a2a-oracle-replay"/os.environ["SLURM_JOB_ID"];out.mkdir(parents=True,exist_ok=True)
report={"completed":False,"training_updates":0,"scope":"CPU physics-only oracle replay","oracle_replays":[]};env=None;start=time.time()
try:
 from libero.libero import benchmark,get_libero_path
 from libero.libero.envs.env_wrapper import ControlEnv
 parent=ROOT/"artifacts/a2a-offline-audit/14467555"
 offline=json.loads((parent/"result.json").read_text());assert offline["completed"]
 payload=parent/"oracle_actions.npz";arrays=np.load(payload,allow_pickle=False)
 report["checkpoint_sha256"]=offline["checkpoint_sha256"];report["oracle_arrays_sha256"]=hashlib.sha256(payload.read_bytes()).hexdigest()
 suite=benchmark.get_benchmark_dict()["libero_object"]();task=suite.get_task(0)
 env=ControlEnv(bddl_file_name=str(Path(get_libero_path("bddl_files"))/task.problem_folder/task.bddl_file),
                use_camera_obs=False,has_renderer=False,has_offscreen_renderer=False)
 report["action_controller"]=type(env.env.robots[0].controller).__name__
 for name in ("demo_28","demo_36"):
  for mode in ("gt","ae"):
   env.seed(0);env.reset();env.set_init_state(arrays[name+"_state0"]);success=False;records=[]
   for step,action in enumerate(arrays[name+"_"+mode]):
    _,reward,done,_=env.step(np.clip(action,-1,1));success=bool(env.check_success())
    assert np.isfinite(env.sim.data.qpos).all()
    records.append({"step":step,"action":action.tolist(),"success":success,"reward":float(reward),
                    "eef_pos":env.env._get_observations()["robot0_eef_pos"].tolist(),
                    "gripper_qpos":env.env._get_observations()["robot0_gripper_qpos"].tolist()})
    if success or done:break
   (out/(name+"_"+mode+".json")).write_text(json.dumps(records)+"\n")
   row={"demo":name,"mode":mode,"success":success,"steps":len(records)}
   report["oracle_replays"].append(row);(out/"result.json").write_text(json.dumps(report,indent=2)+"\n")
   print(json.dumps(row),flush=True)
 report["completed"]=True
 report["interpretation_rule"]="AEfailure only diagnosed when corresponding GTsameinitialstate replay succeeds;future-groundtruth oracle is notdeployablepolicy."
except Exception as exc:
 report["error"]=repr(exc)
 import traceback;traceback.print_exc()
finally:
 if env is not None:env.close()
 report["wall_seconds"]=time.time()-start
 (out/"result.json").write_text(json.dumps(report,indent=2)+"\n")
 print(json.dumps(report,indent=2),flush=True)
sys.exit(0 if report["completed"] else 1)
