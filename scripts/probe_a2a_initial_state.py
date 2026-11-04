"""Frozen A2A initial-state isolation: joint vs grip inputs and physical grip warmup."""
import hashlib,json,os,random,sys,time,traceback
from collections import deque
from pathlib import Path
import h5py,numpy as np,torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from adapters.a2a_model import build_a2a,load_state
from experiments.a2a_run import live_obs
job=os.environ["SLURM_JOB_ID"];out=ROOT/"artifacts/a2a-initial-state-probe"/job;out.mkdir(parents=True,exist_ok=True)
r={"completed":False,"training_updates":0,"scope":"Exploratory two-state300step pilot;not replacement for the primary20rollout protocol",
   "initial_state_ids":[20,38],"settle_steps":10,"rollout_horizon":300,"sampling_steps":6,"execution_horizon":1,
   "input_probes":[],"rollouts":[]}
start=time.time();env=None
def write(): (out/"result.json").write_text(json.dumps(r,indent=2)+"\n")
def digest(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for block in iter(lambda:f.read(8*1024*1024),b""):h.update(block)
 return h.hexdigest()
def tensor_history(obs,device):
 v=live_obs(obs)
 return {k:torch.from_numpy(np.stack([value]*8)).unsqueeze(0).to(device) for k,value in v.items()}
try:
 from libero.libero import benchmark,get_libero_path
 from libero.libero.envs import OffScreenRenderEnv
 assert torch.cuda.is_available();torch.set_num_threads(4);device=torch.device("cuda:0")
 r["gpu"]=torch.cuda.get_device_name(0);policies={};raw_policies={}
 for seed in (0,1):
  m=json.loads((ROOT/("configs/a2a/frozen_causal_seed"+str(seed)+"_10000.json")).read_text())
  path=Path(m["checkpoint"]);assert digest(path)==m["checkpoint_sha256"]
  saved=torch.load(path,map_location="cpu");assert saved["config"]==m["config"] and saved["step"]==10000
  policy=build_a2a(6).to(device);load_state(policy,saved["ema_policy"]);policy.eval()
  policies[seed]=policy;r["seed"+str(seed)+"_checkpoint_sha256"]=m["checkpoint_sha256"]
  raw=build_a2a(6).to(device);load_state(raw,saved["policy"]);raw.eval()
  raw_policies[seed]=raw
  del saved
 n0=policies[0].normalizer.state_dict();n1=policies[1].normalizer.state_dict()
 assert set(n0)==set(n1)
 for key in n0:torch.testing.assert_close(n0[key],n1[key],rtol=0,atol=0)
 split=json.loads((ROOT/"configs/act/split.json").read_text())
 DATA=Path.home()/"scratch/robot_manipulation/datasets/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5"
 with h5py.File(DATA,"r") as f:
  q=np.stack([np.r_[f["data"][name]["obs/joint_states"][0],f["data"][name]["obs/gripper_states"][0]] for name in split["train"]]).astype(np.float32)
 r["training_initial_gripper_min"]=q[:,7:].min(axis=0).tolist()
 r["training_initial_gripper_max"]=q[:,7:].max(axis=0).tolist()
 r["training_initial_gripper_mean"]=q[:,7:].mean(axis=0).tolist()
 q=torch.from_numpy(q).to(device)
 nq=policies[0].normalizer["agent_pos"].normalize(q)
 suite=benchmark.get_benchmark_dict()["libero_object"]();task=suite.get_task(0);states=suite.get_task_init_states(0)
 env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path("bddl_files"))/task.problem_folder/task.bddl_file),
                        camera_heights=128,camera_widths=128,render_gpu_device_id=0)
 with torch.no_grad():
  for initial in (20,38):
   for warm_name,warm_grip in (("original_zero_grip",0.0),("explicit_open_grip",-1.0)):
    for seed,policy in policies.items():
     random.seed(initial);np.random.seed(initial);torch.manual_seed(initial);torch.cuda.manual_seed_all(initial)
     env.seed(initial);env.reset();obs=env.set_init_state(states[initial])
     before_settle=np.r_[obs["robot0_joint_pos"],obs["robot0_gripper_qpos"]]
     for _ in range(10):
      a=np.zeros(7);a[-1]=warm_grip;obs,_,_,_=env.step(a)
     context=tensor_history(obs,device)
     current=context["agent_pos"][0,-1];norm=policies[0].normalizer["agent_pos"].normalize(current)
     nearest=int(torch.argmin(torch.linalg.vector_norm(nq-norm,dim=-1)))
     delta=norm-nq[nearest]
     warm_state={"initial_state_id":initial,"warmup":warm_name,"seed":seed,
       "before_settle_proprio":before_settle.tolist(),"after_settle_proprio":current.cpu().tolist(),
       "nearest_expert_episode":split["train"][nearest],"nearest_expert_proprio":q[nearest].cpu().tolist(),
       "normalized_arm_distance":float(torch.linalg.vector_norm(delta[:7])),
       "normalized_grip_distance":float(torch.linalg.vector_norm(delta[7:]))}
     if seed==0:
      contexts={"live":context}
      for label,part in (("expert_arm_only",slice(0,7)),("expert_grip_only",slice(7,9)),("expert_all_proprio",slice(0,9))):
       newq=context["agent_pos"].clone();newq[...,part]=q[nearest,part]
       contexts[label]={k:(newq if k=="agent_pos" else v) for k,v in context.items()}
      probe=dict(warm_state,predictions={})
      for name,inputs in contexts.items():
       probe["predictions"][name]={}
       for model_seed,model in policies.items():
        torch.manual_seed(initial);torch.cuda.manual_seed_all(initial)
        action=model.predict_action(inputs)["action"][0,0]
        probe["predictions"][name][str(model_seed)]=action.cpu().tolist()
      probe["raw_weight_live_predictions"]={}
      for model_seed,raw_model in raw_policies.items():
       torch.manual_seed(initial);torch.cuda.manual_seed_all(initial)
       probe["raw_weight_live_predictions"][str(model_seed)]=raw_model.predict_action(context)["action"][0,0].cpu().tolist()
      r["input_probes"].append(probe);write()
     history=deque([live_obs(obs)]*8,maxlen=8)
     frames=[Image.fromarray(obs["agentview_image"][::-1])];actions=[];success=False
     for step in range(300):
      b={k:torch.from_numpy(np.stack([item[k] for item in history])).unsqueeze(0).to(device) for k in history[0]}
      predicted=policy.predict_action(b)["action"][0,0].cpu().numpy()
      assert np.isfinite(predicted).all()
      executed=np.clip(predicted,-1,1)
      obs,reward,done,_=env.step(executed)
      assert np.isfinite(env.sim.data.qpos).all()
      success=bool(env.check_success());actions.append(executed.tolist());history.append(live_obs(obs))
      if (step+1)%5==0:frames.append(Image.fromarray(obs["agentview_image"][::-1]))
      if success or done:break
     frames.append(Image.fromarray(obs["agentview_image"][::-1]))
     label="init"+str(initial)+"_"+warm_name+"_seed"+str(seed)
     frames[0].save(out/(label+".gif"),save_all=True,append_images=frames[1:],duration=250,loop=0)
     (out/(label+"_actions.json")).write_text(json.dumps(actions)+"\n")
     row=dict(warm_state,success=success,steps=len(actions),completed=True)
     r["rollouts"].append(row);write();print(json.dumps(row),flush=True)
 r["completed"]=True
 r["interpretation"]="Input substitutions are off-manifold sensitivity probes. Warmup changes actual initial gripper command/feedback/images;the two-state pilot cannot replace fixed-protocol20trial success or establish broad robustness."
except Exception as exc:
 r["error"]=repr(exc);traceback.print_exc()
finally:
 if env is not None:env.close()
 r["wall_seconds"]=time.time()-start;write();print(json.dumps({k:v for k,v in r.items() if k!="input_probes"},indent=2),flush=True)
sys.exit(0 if r["completed"] else 1)
