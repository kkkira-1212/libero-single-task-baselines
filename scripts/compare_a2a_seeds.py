"""Same-input frozen A2A seed comparison. No optimizer, backward or training updates."""
import hashlib,json,os,random,sys,time,traceback
from pathlib import Path
import h5py,numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from adapters.a2a_model import build_a2a,load_state
from adapters.libero_causal_contract import sample_options,CONTRACT_VERSION
from experiments.a2a_run import CachedSamples
from roboverse_learn.il.utils.pytorch_util import dict_apply

job=os.environ["SLURM_JOB_ID"];out=ROOT/"artifacts/a2a-seed-comparison"/job
out.mkdir(parents=True,exist_ok=True)
r={"completed":False,"training_updates":0,"purpose":"same-input raw/EMA AE/flow and verified sampling comparison","models":{}}
start=time.time()
cpu_fast=os.environ.get("A2A_COMPARE_CPU_FAST")=="1"
def write(): (out/"result.json").write_text(json.dumps(r,indent=2)+"\n")
def digest(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
 return h.hexdigest()
def batch(ds,ids,device):
 items=[ds[int(i)] for i in ids]
 return {"obs":{k:torch.from_numpy(np.stack([s["obs"][k] for s in items])).to(device) for k in items[0]["obs"]},
         "action":torch.from_numpy(np.stack([s["action"] for s in items])).to(device)}
def plan(ds):
 ids=[];meta=[];offset=0
 for ep,d in enumerate(ds.data):
  a=d["actions"];n=len(a)
  switches=np.flatnonzero(np.sign(a[1:,6])!=np.sign(a[:-1,6]))
  anchors=sorted(set([0,(n-2)//3,2*(n-2)//3,n-2]+switches.tolist()))
  for t in anchors:
   ids.append(offset+t)
   meta.append({"episode":ds.episodes[ep],"t":t,"target_first":t+1,
                "gripper_switch":bool(np.sign(a[t+1,6])!=np.sign(a[t,6]))})
  offset+=n-1
 return ids,meta
def metrics(pred,target,meta):
 diff=pred-target;first=diff[:,0];sign=np.sign(pred[:,0,6]);truth=np.sign(target[:,0,6])
 switch=np.array([x["gripper_switch"] for x in meta],dtype=bool)
 def acc(mask):return float(np.mean(sign[mask]==truth[mask])) if mask.any() else None
 return {"samples":len(pred),"first_translation_rmse":float(np.sqrt(np.mean(first[:,:3]**2))),
  "first_rotation_rmse":float(np.sqrt(np.mean(first[:,3:6]**2))),
  "first_action_rmse":float(np.sqrt(np.mean(first**2))),
  "per_dimension_mse":np.mean(diff**2,axis=(0,1)).tolist(),
  "first_gripper_mae":float(np.mean(np.abs(first[:,6]))),
  "first_gripper_sign_accuracy":acc(np.ones(len(pred),dtype=bool)),
  "first_gripper_sign_accuracy_positive":acc(truth>0),"first_gripper_sign_accuracy_negative":acc(truth<0),
  "switch_samples":int(switch.sum()),"switch_first_gripper_sign_accuracy":acc(switch),
  "predicted_std":np.std(pred,axis=(0,1)).tolist(),"target_std":np.std(target,axis=(0,1)).tolist(),
  "raw_outside_control_limits_fraction":float(np.mean(np.abs(pred)>1))}
def forward(policy,b,check=False):
 nobs=policy.normalizer.normalize(b["obs"])
 actions=policy.normalizer["action"].normalize(b["action"])[:,7:15]
 bs=len(actions);flat=dict_apply(nobs,lambda x:x[:,:8].reshape(-1,*x.shape[2:]))
 cond=policy.obs_projector(policy.obs_encoder(flat).reshape(bs,-1))
 z0=policy.history_action_encoder(nobs["agent_pos"])
 z1=policy.action_encoder(actions)
 ae=policy.normalizer["action"].unnormalize(policy.action_decoder(z1))
 zends={};pred={}
 for steps in (1,6):
  zends[steps]=policy.flow_matcher.sample(policy.flow_net,shape=(bs,policy.latent_dim),
      device=policy.device,start=z0,num_steps=steps,global_cond=cond)
  pred[steps]=policy.normalizer["action"].unnormalize(policy.action_decoder(zends[steps]))
 if check:
  policy.num_sampling_steps=6
  actual=policy.predict_action(b["obs"])["action"]
  torch.testing.assert_close(actual,pred[6],rtol=1e-5,atol=1e-6)
  torch.manual_seed(12345)
  fm,_=policy.flow_matcher.compute_loss(policy.flow_net,target=z1,start=z0,global_cond=cond)
  components={"flow_MSE":float(fm),"AE_normalized_L1":float(torch.nn.functional.l1_loss(policy.action_decoder(z1),actions)),
   "flow6_normalized_L1":float(torch.nn.functional.l1_loss(policy.action_decoder(zends[6]),actions)),
   "latent_consistency_MSE":float(torch.nn.functional.mse_loss(zends[6],z1))}
  manual=fm+policy.action_ae["enc_recon_weight"]*components["AE_normalized_L1"]+policy.action_ae["flow_recon_weight"]*components["flow6_normalized_L1"]+policy.consistency_weight*components["latent_consistency_MSE"]
  torch.manual_seed(12345);author=policy.compute_loss(b)
  torch.testing.assert_close(author,manual,rtol=1e-5,atol=1e-6)
  components["manual_total"]=float(manual);components["author_total"]=float(author)
 else:components=None
 arrays={k:v.detach().cpu().numpy() for k,v in {"target":b["action"][:,7:15],"ae":ae,"flow1":pred[1],"flow6":pred[6],
       "z0":z0,"z1":z1,"z6":zends[6]}.items()}
 return arrays,components
def live_initial_probe(policies,datasets,device):
 from libero.libero import benchmark,get_libero_path
 from libero.libero.envs import OffScreenRenderEnv
 from experiments.a2a_run import live_obs
 suite=benchmark.get_benchmark_dict()["libero_object"]();task=suite.get_task(0)
 states=suite.get_task_init_states(0)
 env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path("bddl_files"))/task.problem_folder/task.bddl_file),
    camera_heights=128,camera_widths=128,render_gpu_device_id=0)
 results=[];train=datasets["train"]
 rawq=np.stack([np.r_[d["obs/joint_states"][0],d["obs/gripper_states"][0]] for d in train.data]).astype(np.float32)
 with torch.no_grad():expertq=policies[0].normalizer["agent_pos"].normalize(torch.from_numpy(rawq).to(device))
 try:
  with torch.no_grad():
   for state in (20,38):
    np.random.seed(state);random.seed(state);torch.manual_seed(state);torch.cuda.manual_seed_all(state)
    env.seed(state);env.reset();obs=env.set_init_state(states[state])
    for _ in range(10):obs,_,_,_=env.step(np.zeros(7))
    live=live_obs(obs)
    lb={k:torch.from_numpy(np.stack([value]*8)).unsqueeze(0).to(device) for k,value in live.items()}
    lq=policies[0].normalizer["agent_pos"].normalize(lb["agent_pos"][0,-1])
    dist=torch.linalg.vector_norm(expertq-lq,dim=-1);nearest=int(torch.argmin(dist))
    index=0 if nearest==0 else int(train.ends[nearest-1])
    expert=batch(train,[index],device)["obs"]
    contexts={"same_live_observation":lb,"nearest_expert_initial_observation":expert,
      "expert_images_live_proprio":{k:(lb[k] if k=="agent_pos" else expert[k]) for k in lb},
      "live_images_expert_proprio":{k:(expert[k] if k=="agent_pos" else lb[k]) for k in lb}}
    row={"initial_state_id":state,"nearest_expert_episode":train.episodes[nearest],
      "nearest_initial_proprio_distance_normalized":float(dist[nearest]),
      "live_proprio_outside_training_limits_axes":int((lq.abs()>1).sum()),"contexts":{}}
    for name,context in contexts.items():
     predictions={}
     for seed,policy in policies.items():
      torch.manual_seed(state);torch.cuda.manual_seed_all(state);policy.num_sampling_steps=6
      prediction=policy.predict_action(context)["action"][0,0]
      predictions[str(seed)]={"raw_action":prediction.cpu().tolist(),
                             "executed_action":prediction.clamp(-1,1).cpu().tolist()}
     row["contexts"][name]=predictions
    results.append(row)
 finally:env.close()
 return {"cases":results,"limits":"Input hybrids are off-manifold diagnostic counterfactuals,not task-success evidence or proof of a unique cause."}

try:
 torch.set_num_threads(4)
 if not cpu_fast:assert torch.cuda.is_available()
 device=torch.device("cpu" if cpu_fast else "cuda:0")
 r["gpu"]="CPU" if cpu_fast else torch.cuda.get_device_name(0)
 r["cpu_preliminary"]=cpu_fast
 manifests={seed:json.loads((ROOT/("configs/a2a/frozen_causal_seed"+str(seed)+"_10000.json")).read_text()) for seed in (0,1)}
 configs={seed:m["config"] for seed,m in manifests.items()}
 differences={k:[configs[0].get(k),configs[1].get(k)] for k in set(configs[0])|set(configs[1]) if configs[0].get(k)!=configs[1].get(k)}
 assert set(differences)=={"seed","purpose"}
 assert configs[0]["sample_contract"]==CONTRACT_VERSION
 r["config_differences"]=differences
 sources=("a2a_run.py","a2a_model.py","a2a_policy.py","libero_sequence.py","libero_causal_contract.py")
 hashes={}
 for seed,m in manifests.items():
  folder=ROOT/"artifacts/a2a-train"/m["training_job"]
  hashes[str(seed)]={name:digest(folder/name) for name in sources}
 assert hashes["0"]==hashes["1"],"Actual training source snapshots differ"
 r["identical_training_source_sha256"]=hashes["0"]
 split=json.loads((ROOT/"configs/act/split.json").read_text())
 DATA=Path.home()/"scratch/robot_manipulation/datasets/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5"
 assert digest(DATA)==configs[0]["dataset_sha256"]
 datasets={s:CachedSamples(split[s],**sample_options(configs[0])) for s in ("train","validation")}
 plans={s:plan(ds) for s,ds in datasets.items()}
 if cpu_fast:
  for subset,(ids,meta) in list(plans.items()):
   take=np.linspace(0,len(ids)-1,32 if subset=="train" else 16,dtype=int)
   plans[subset]=([ids[i] for i in take],[meta[i] for i in take])
 r["sample_plan"]={s:meta for s,(ids,meta) in plans.items()}
 r["sampling_reconstruction"]={}
 train=datasets["train"]
 first_actions=np.concatenate([d["actions"][1:] for d in train.data])
 switches=np.concatenate([np.sign(d["actions"][1:,6])!=np.sign(d["actions"][:-1,6]) for d in train.data])
 for seed,m in manifests.items():
  actual=json.loads((ROOT/"artifacts/a2a-train"/m["training_job"]/"result.json").read_text())["sample_order_sha256"]
  draws=np.random.RandomState(seed).randint(len(train),size=configs[seed]["updates"]*configs[seed]["batch_size"])
  verified=hashlib.sha256(draws.astype("<i8").tobytes()).hexdigest()==actual
  item={"actual_sample_order_sha256":actual,"reconstructed_sha_matches":verified}
  if verified:
   ep=np.searchsorted(train.ends,draws,side="right")
   item.update(total_samples=len(draws),unique_anchor_coverage=float(len(np.unique(draws))/len(train)),
    gripper_switch_draws=int(switches[draws].sum()),positive_gripper_fraction=float(np.mean(first_actions[draws,6]>0)),
    draws_per_training_episode={name:int((ep==i).sum()) for i,name in enumerate(train.episodes)})
  r["sampling_reconstruction"][str(seed)]=item
 print("Actual sources/config/data/sample order compared",flush=True);write()
 normalizers={};output_arrays={};live_policies={}
 for seed,m in manifests.items():
  checkpoint=Path(m["checkpoint"]);assert digest(checkpoint)==m["checkpoint_sha256"]
  saved=torch.load(checkpoint,map_location="cpu")
  assert saved["step"]==10000 and saved["config"]==configs[seed] and saved["split"]==split
  assert saved["source_commit"]=="a5792ecf4e7f8fa4d85fe66ea9a50618138f925c"
  r["models"][str(seed)]={"checkpoint_sha256":m["checkpoint_sha256"],"weights":{}}
  for weight in (("ema_policy",) if cpu_fast else ("policy","ema_policy")):
   label=str(seed)+"_"+weight
   normalizers[label]={k:v.detach().cpu().numpy() for k,v in saved[weight].items() if k.startswith("normalizer.")}
   torch.manual_seed(112233)
   policy=build_a2a(6).to(device);load_state(policy,saved[weight]);policy.eval()
   wr={}
   with torch.no_grad():
    for subset,ds in datasets.items():
     ids,meta=plans[subset];collected={};components=None
     for off in range(0,len(ids),8):
      arrays,component=forward(policy,batch(ds,ids[off:off+8],device),check=(off==0))
      if component:components=component
      for k,value in arrays.items():collected.setdefault(k,[]).append(value)
     vals={k:np.concatenate(v) for k,v in collected.items()}
     results={name:metrics(vals[name],vals["target"],meta) for name in ("ae","flow1","flow6")}
     zerr=vals["z6"]-vals["z1"]
     results["latent_within_model"]={"history_std":float(vals["z0"].std(axis=0).mean()),
       "future_std":float(vals["z1"].std(axis=0).mean()),"flow6_std":float(vals["z6"].std(axis=0).mean()),
       "flow6_target_mse":float(np.mean(zerr**2)),
       "note":"Latent bases differ across checkpoints;do not swap encoders/decoders across seeds."}
     results["first_batch_loss_components"]=components
     wr[subset]=results
     for name in ("target","ae","flow1","flow6"):output_arrays[label+"_"+subset+"_"+name]=vals[name]
   r["models"][str(seed)]["weights"][weight]=wr
   write();print(json.dumps({"seed":seed,"weights":weight,"train_windows":len(plans["train"][0]),"validation_windows":len(plans["validation"][0])}),flush=True)
   # Future-ground-truth AE / expert-observation flow replay arrays are diagnostic oracles.
   if weight=="ema_policy" and not cpu_fast:
    with h5py.File(DATA,"r") as f:
     for name in ("demo_28","demo_36"):
      d=f["data"][name];gt=d["actions"][:].astype(np.float32);ds=CachedSamples([name],**sample_options(configs[seed]))
      ae=[];flow=[]
      with torch.no_grad():
       for off in range(0,len(ds),8):
        vals,_=forward(policy,batch(ds,np.arange(off,min(off+8,len(ds))),device))
        ae.append(vals["ae"][:,0]);flow.append(vals["flow6"][:,0])
      output_arrays["seed"+str(seed)+"_"+name+"_gt"]=gt
      output_arrays["seed"+str(seed)+"_"+name+"_ae"]=np.concatenate([gt[:1],np.concatenate(ae)])
      output_arrays["seed"+str(seed)+"_"+name+"_flow6"]=np.concatenate([gt[:1],np.concatenate(flow)])
      output_arrays[name+"_state0"]=d["states"][0]
   if weight=="ema_policy" and not cpu_fast:live_policies[seed]=policy
   else:del policy
   if not cpu_fast:torch.cuda.empty_cache()
  del saved
 if not cpu_fast:
  try:
   r["same_live_initial_observation_probe"]=live_initial_probe(live_policies,datasets,device)
   r["live_probe_completed"]=True
  except Exception as probe_error:
   r["live_probe_completed"]=False;r["live_probe_error"]=repr(probe_error);traceback.print_exc()
 baseline=normalizers["0_ema_policy"]
 r["normalizer_max_abs_difference_vs_seed0_EMA"]={
  label:{key:float(np.max(np.abs(value-baseline[key]))) for key,value in stats.items()}
  for label,stats in normalizers.items()}
 np.savez_compressed(out/"comparison_arrays.npz",**output_arrays)
 r["arrays_sha256"]=digest(out/"comparison_arrays.npz")
 r["scope_limits"]="Same chosen expert windows/oracle arrays,not full-dataset errors or deployable oracle success. No training updates;training hardware vs seed not isolated."
 r["completed"]=True
except Exception as exc:
 r["error"]=repr(exc);traceback.print_exc()
finally:
 r["wall_seconds"]=time.time()-start;write();print(json.dumps({k:v for k,v in r.items() if k not in ("sample_plan","models")},indent=2),flush=True)
sys.exit(0 if r["completed"] else 1)
