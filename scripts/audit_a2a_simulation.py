"""Zero-update A2A audit: loss/grad parity, dense AE errors, oracle replay, execution probe."""
import hashlib,json,os,sys,time
from collections import deque
from pathlib import Path
import h5py,numpy as np,torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from adapters.a2a_model import build_a2a,load_state
from experiments.a2a_run import CachedSamples,batch_from,live_obs
from roboverse_learn.il.utils.pytorch_util import dict_apply
job=os.environ["SLURM_JOB_ID"];out=ROOT/"artifacts/a2a-audit"/job;out.mkdir(parents=True,exist_ok=True)
report={"completed":False,"training_updates":0,"oracle_replays":[],"execution_probes":[]};env=None;start=time.time()
def write(): (out/"result.json").write_text(json.dumps(report,indent=2)+"\n")
def digest(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
 return h.hexdigest()
def errors(pred,target):
 e=pred-target
 return {"mse_per_dimension":np.mean(e*e,axis=tuple(range(e.ndim-1))).tolist(),
         "gripper_sign_accuracy":float(np.mean(np.sign(pred[...,6])==np.sign(target[...,6])))}
try:
 torch.set_num_threads(4);torch.manual_seed(0);np.random.seed(0)
 train=json.loads((ROOT/"artifacts/a2a-train/14463324/result.json").read_text())
 checkpoint=Path(train["checkpoint"]);report["checkpoint_sha256"]=digest(checkpoint)
 old=json.loads((ROOT/"artifacts/a2a-eval/14466095/result.json").read_text())
 assert report["checkpoint_sha256"]==old["checkpoint_sha256"]
 saved=torch.load(checkpoint,map_location="cpu")
 assert saved["step"]==40000
 policy=build_a2a(6).cuda();load_state(policy,saved["ema_policy"]);policy.eval();split=saved["split"];del saved
 cpu_dir=ROOT/"artifacts/a2a-offline-audit"/sys.argv[1]
 offline=json.loads((cpu_dir/"result.json").read_text())
 assert offline["completed"] and offline["checkpoint_sha256"]==report["checkpoint_sha256"]
 report["offline_parent_job"]=sys.argv[1]
 report["offline_findings"]={k:v for k,v in offline.items() if k not in ("completed","wall_seconds","oracle_replays","execution_probes")}
 cached={}
 arrays=np.load(cpu_dir/"oracle_actions.npz",allow_pickle=False)
 for name in ("demo_28","demo_36"):
  cached[name]={key:arrays[name+"_"+key] for key in ("gt","ae","state0")}
 from libero.libero import benchmark,get_libero_path
 from libero.libero.envs import OffScreenRenderEnv
 suite=benchmark.get_benchmark_dict()["libero_object"]();task=suite.get_task(0)
 env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path("bddl_files"))/task.problem_folder/task.bddl_file),
                       camera_heights=128,camera_widths=128,render_gpu_device_id=0)
 for demo,content in cached.items():
  for mode in ("gt","ae"):
   env.seed(0);env.reset();obs=env.set_init_state(content["state0"]);frames=[Image.fromarray(obs["agentview_image"][::-1])];success=False;traj=[]
   for step,action in enumerate(content[mode]):
    obs,_,done,_=env.step(np.clip(action,-1,1));success=bool(env.check_success());traj.append(np.clip(action,-1,1).tolist())
    if step%5==0:frames.append(Image.fromarray(obs["agentview_image"][::-1]))
    if success or done:break
   frames[0].save(out/(demo+"_"+mode+".gif"),save_all=True,append_images=frames[1:],duration=250,loop=0)
   report["oracle_replays"].append({"demo":demo,"mode":mode,"success":success,"steps":len(traj),
       "validity":"AE comparison only interpretable when same-demo GT replay succeeds;state-only restore,no nativeXML replay"})
   write()
 # Original model unchanged; short closed-loop counterfactual only changes chunk execution count.
 for initial in (20,38):
  for execute in (1,8):
   env.seed(initial);env.reset();obs=env.set_init_state(suite.get_task_init_states(0)[initial])
   for _ in range(10):obs,_,_,_=env.step(np.zeros(7))
   history=deque([live_obs(obs)]*8,maxlen=8);frames=[Image.fromarray(obs["agentview_image"][::-1])]
   count=0;success=False;done=False
   with torch.no_grad():
    while count<300 and not success and not done:
     b={k:torch.from_numpy(np.stack([x[k] for x in history])).unsqueeze(0).cuda() for k in history[0]}
     chunk=policy.predict_action(b)["action"][0].cpu().numpy()
     for action in chunk[:execute]:
      obs,_,done,_=env.step(np.clip(action,-1,1));count+=1;history.append(live_obs(obs));success=bool(env.check_success())
      assert np.isfinite(env.sim.data.qpos).all()
      if count%5==0:frames.append(Image.fromarray(obs["agentview_image"][::-1]))
      if count>=300 or success or done:break
   frames[0].save(out/("init%d_exec%d.gif"%(initial,execute)),save_all=True,append_images=frames[1:],duration=250,loop=0)
   report["execution_probes"].append({"initial_id":initial,"execute":execute,"steps":count,"success":success,"horizon":300})
   write()
 report["completed"]=True
except Exception as exc:
 report["error"]=repr(exc)
 import traceback;traceback.print_exc()
finally:
 if env is not None:env.close()
 report["wall_seconds"]=time.time()-start;write()
 print(json.dumps({k:v for k,v in report.items() if k!="dense_AE"},indent=2),flush=True)
sys.exit(0 if report["completed"] else 1)
