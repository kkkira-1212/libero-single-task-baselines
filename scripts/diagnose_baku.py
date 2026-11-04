"""Frozen BAKU diagnosis: teacher-forced errors, input-path parity, short aggregation probe."""
import json,os,pickle,random,sys,time
from pathlib import Path
import h5py,numpy as np,torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"third_party/BAKU/baku"))
from agent.baku import BCAgent
from adapters.baku_live import online_observation,identity_stats
job=os.environ["SLURM_JOB_ID"];out=ROOT/"artifacts/baku-diagnosis"/job;out.mkdir(parents=True,exist_ok=True)
report={"completed":False,"training_updates":0,"short_rollouts":[]};start=time.time();env=None
try:
 torch.set_num_threads(4);random.seed(0);np.random.seed(0);torch.manual_seed(0)
 checkpoint=Path.home()/"scratch/robot_manipulation/baku-training/14386819/latest.pt"
 saved=torch.load(checkpoint,map_location="cpu");c=saved["config"]
 assert saved["step"]==3000 and saved["source_commit"]=="47e367059191de30ac4b6353ff5615fdef11d8c5"
 agent=BCAgent(**c["model_config"]);agent.load_snapshot(saved["snapshot"],eval=True);del saved
 report.update(checkpoint=str(checkpoint),gpu=torch.cuda.get_device_name(0))
 captured=[]
 def hook(module,inputs,output):
  dist=output[0] if isinstance(output,tuple) else output
  captured.append(dist.mean.detach().cpu().numpy())
 handle=agent.actor.register_forward_hook(hook)
 parity=[];teacher={}
 for subset in ("train","validation"):
  task="pick_up_the_alphabet_soup_and_place_it_in_the_basket"
  with (Path(c["data_root"])/subset/"libero_object"/(task+".pkl")).open("rb") as f:data=pickle.load(f)
  errors=[];truths=[];signs=[];records=[]
  for ep in range(4):
   obs=data["observations"][ep];actions=data["actions"][ep]
   for t in np.linspace(0,len(actions)-11,8,dtype=int):
    batch={key:torch.from_numpy(np.ascontiguousarray(obs[key][t].transpose(2,0,1)).astype(np.float32)/255)[None,None] for key in ("pixels","pixels_egocentric")}
    state=np.concatenate([obs["joint_states"][t],obs["gripper_states"][t]]).astype(np.float32)
    batch["proprioceptive"]=torch.from_numpy(state/(1+1e-5))[None,None]
    target=actions[t:t+10].astype(np.float32)/(1+1e-5)
    batch["actions"]=torch.from_numpy(target)[None,None]
    batch["task_emb"]=torch.from_numpy(data["task_emb"])[None]
    with torch.no_grad():
     metrics=agent.update(iter([batch]),0,update=False)
    predicted=metrics["pred_action"].reshape(10,7)
    errors.append((predicted-target)**2);truths.append(target)
    signs.append(np.sign(predicted[:,6])==np.sign(target[:,6]))
    records.append({"episode_id":data["episode_ids"][ep],"t":int(t),
                    "first_predicted_action":predicted[0].tolist(),"first_target_action":target[0].tolist()})
    if ep==0 and t==0:
     offline=predicted.copy();captured.clear();agent.buffer_reset()
     live={key:np.ascontiguousarray(obs[key][t].transpose(2,0,1)) for key in ("pixels","pixels_egocentric")}
     live["proprioceptive"]=state.copy()
     with torch.no_grad():action=agent.act(live,{"task_emb":data["task_emb"]},identity_stats(),0,0,eval_mode=True)
     online=captured[-1].reshape(10,7)
     parity.append({"subset":subset,"raw_chunk_max_difference":float(np.max(np.abs(offline-online))),
                    "executed_first_action_max_difference":float(np.max(np.abs(offline[0]-action)))})
    captured.clear()
  err=np.stack(errors);target=np.stack(truths)
  teacher[subset]={"samples":len(errors),"mse_per_dimension":err.mean((0,1)).tolist(),
                  "first_action_rmse":float(np.sqrt(err[:,0].mean())),
                  "gripper_sign_accuracy":float(np.mean(signs)),
                  "target_gripper_positive_fraction":float(np.mean(target[:,:,6]>0)),
                  "records":records}
  del data
 report["teacher_forced"]=teacher;report["input_path_parity"]=parity
 write=lambda: (out/"result.json").write_text(json.dumps(report,indent=2)+"\n")
 write();print("Teacher forcing and parity completed",flush=True)
 from libero.libero import benchmark,get_libero_path
 from libero.libero.envs import OffScreenRenderEnv
 suite=benchmark.get_benchmark_dict()["libero_object"]();task=suite.get_task(0);states=suite.get_task_init_states(0)
 env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path("bddl_files"))/task.problem_folder/task.bddl_file),
                       camera_heights=128,camera_widths=128,render_gpu_device_id=0)
 prompt={"task_emb":np.load(Path(c["data_root"])/"task_embedding.npy")}
 for initial_id in (20,38):
  for mode in ("official_aggregation","latest_chunk_first_action"):
   random.seed(initial_id);np.random.seed(initial_id);torch.manual_seed(initial_id)
   env.seed(initial_id);env.reset();obs=env.set_init_state(states[initial_id])
   for _ in range(10):obs,_,_,_=env.step(np.zeros(7))
   agent.buffer_reset();frames=[Image.fromarray(obs["agentview_image"][::-1])];trajectory=[]
   with torch.no_grad():
    for step in range(180):
     captured.clear()
     aggregated=agent.act(online_observation(obs),prompt,identity_stats(),step,step,eval_mode=True)
     latest=captured[-1].reshape(-1,10,7)[-1,0]
     action=aggregated if mode=="official_aggregation" else latest
     assert action.shape==(7,) and np.isfinite(action).all()
     obs,reward,done,_=env.step(np.clip(action,-1,1))
     success=bool(env.check_success())
     trajectory.append({"step":step,"aggregated":aggregated.tolist(),"latest":latest.tolist(),
                        "executed":np.clip(action,-1,1).tolist(),"success":success})
     if (step+1)%5==0:frames.append(Image.fromarray(obs["agentview_image"][::-1]))
     if success or done:break
   name="%d_%s"%(initial_id,mode)
   frames[0].save(out/(name+".gif"),save_all=True,append_images=frames[1:],duration=250,loop=0)
   (out/(name+".json")).write_text(json.dumps(trajectory))
   report["short_rollouts"].append({"initial_id":initial_id,"mode":mode,"steps":len(trajectory),"success":success})
   write()
 handle.remove()
 report["completed"]=True
except Exception as exc:
 report["error"]=repr(exc)
 import traceback;traceback.print_exc()
finally:
 if env is not None:env.close()
 report["wall_seconds"]=time.time()-start
 (out/"result.json").write_text(json.dumps(report,indent=2)+"\n")
 print(json.dumps({k:v for k,v in report.items() if k!="teacher_forced"},indent=2),flush=True)
sys.exit(0 if report["completed"] else 1)
