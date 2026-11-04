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
 ds=CachedSamples(split["train"]);batch=batch_from(ds,[64,200,400,600])
 # Numerically rebuild exactly the author loss with component logging; no optimizer exists.
 nobs=policy.normalizer.normalize(batch["obs"]);a=policy.normalizer["action"].normalize(batch["action"])[:,7:15]
 flat=dict_apply(nobs,lambda x:x[:,:8].reshape(-1,*x.shape[2:]))
 cond=policy.obs_projector(policy.obs_encoder(flat).reshape(4,-1))
 z0=policy.history_action_encoder(nobs["agent_pos"]);z1=policy.action_encoder(a)
 torch.manual_seed(12345)
 fm,_=policy.flow_matcher.compute_loss(policy.flow_net,target=z1,start=z0,global_cond=cond)
 end=policy.flow_matcher.sample(policy.flow_net,shape=(4,512),device=policy.device,start=z0,num_steps=6,global_cond=cond)
 ae=torch.nn.functional.l1_loss(policy.action_decoder(z1),a)
 recon=torch.nn.functional.l1_loss(policy.action_decoder(end),a)
 latent_mse=torch.nn.functional.mse_loss(end,z1);latent_l1=torch.nn.functional.l1_loss(end,z1)
 total=fm+0.5*ae+latent_mse+0.5*recon
 torch.manual_seed(12345);author=policy.compute_loss(batch)
 torch.testing.assert_close(total,author,rtol=1e-5,atol=1e-6)
 report["loss_components"]={k:float(v.detach()) for k,v in {"flow_MSE":fm,"AE_L1":ae,"latent_consistency_MSE":latent_mse,"paper_latent_consistency_L1":latent_l1,"flow_action_L1":recon,"weighted_total":total,"author_total":author}.items()}
 params=[p for p in policy.parameters() if p.requires_grad]
 grads=torch.autograd.grad(total,params,allow_unused=True)
 gradient_map={id(p):g for p,g in zip(params,grads)}
 norms={}
 for name in ("obs_encoder","obs_projector","history_action_encoder","action_encoder","action_decoder","flow_net"):
  gs=[gradient_map.get(id(p)) for p in getattr(policy,name).parameters()]
  gs=[g for g in gs if g is not None]
  norms[name]={"tensors_with_gradient":len(gs),"finite":all(torch.isfinite(g).all().item() for g in gs),
               "L2norm":float(torch.sqrt(sum((g.float()**2).sum() for g in gs))) if gs else 0.0}
 report["gradient_paths"]=norms
 del total,author,fm,ae,recon,latent_mse,latent_l1,cond,z0,z1,end,grads,gradient_map,params
 # Genuine offline adapter vs online-history builder on the same stored8 frames.
 d=ds.data[0];history=[]
 for t in range(57,65):
  fake={"agentview_image":d["obs/agentview_rgb"][t],"robot0_eye_in_hand_image":d["obs/eye_in_hand_rgb"][t],
        "robot0_joint_pos":d["obs/joint_states"][t],"robot0_gripper_qpos":d["obs/gripper_states"][t]}
  history.append(live_obs(fake))
 online={k:torch.from_numpy(np.stack([x[k] for x in history])).unsqueeze(0).cuda() for k in history[0]}
 offline={k:torch.from_numpy(ds[64]["obs"][k]).unsqueeze(0).cuda() for k in history[0]}
 with torch.no_grad():
  p0=policy.predict_action(offline)["action"];p1=policy.predict_action(online)["action"]
 report["offline_online_identical_window_max_difference"]=float((p0-p1).abs().max())
 with torch.no_grad():
  base=policy.predict_action(batch["obs"])["action"]
  perm={k:(v.flip(0) if k in ("agentview","wrist") else v) for k,v in batch["obs"].items()}
  reverse={k:(v.flip(1) if k=="agent_pos" else v) for k,v in batch["obs"].items()}
  report["sensitivity"]={"image_permutation_action_mean_abs_change":float((base-policy.predict_action(perm)["action"]).abs().mean()),
                          "history_reversal_action_mean_abs_change":float((base-policy.predict_action(reverse)["action"]).abs().mean())}
 del ds,batch,offline,online,nobs,a,flat
 datafile=Path.home()/"scratch/robot_manipulation/datasets/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5"
 cached={};report["dense_AE"]={}
 with h5py.File(datafile,"r") as f:
  for name in ("demo_28","demo_36"):
   d=f["data"][name];gt=d["actions"][:].astype(np.float32);tlen=len(gt)
   ids=np.clip(np.arange(tlen)[:,None]+np.arange(8)[None],0,tlen-1)
   recon=[]
   with torch.no_grad():
    for offset in range(0,tlen,16):
     target=torch.from_numpy(gt[ids[offset:offset+16]]).cuda()
     normalized=policy.normalizer["action"].normalize(target)
     predicted=policy.normalizer["action"].unnormalize(policy.action_decoder(policy.action_encoder(normalized)))
     recon.append(predicted.cpu().numpy())
   reconstructed=np.concatenate(recon)
   expected=gt[ids]
   per_horizon=[errors(reconstructed[:,i],expected[:,i]) for i in range(8)]
   switch=np.r_[False,np.sign(gt[1:,6])!=np.sign(gt[:-1,6])]
   report["dense_AE"][name]={"length":tlen,"per_horizon":per_horizon,
                            "first_action_switch_frame_count":int(switch.sum()),
                            "first_action_switch_gripper_sign_accuracy":float(np.mean(np.sign(reconstructed[switch,0,6])==np.sign(gt[switch,6]))) if switch.any() else None}
   cached[name]={"gt":gt,"ae":reconstructed[:,0],"state0":d["states"][0]}
 write();print("Loss/gradient/parity/dense AE audit complete",flush=True)
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
