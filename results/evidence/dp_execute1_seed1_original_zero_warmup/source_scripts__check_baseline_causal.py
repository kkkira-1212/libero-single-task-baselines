"""Real-HDF5 causal baseline gates; CUDA update/reload only after data checks."""
import argparse,hashlib,json,os,random,sys,time,traceback
from pathlib import Path
import h5py,numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from adapters.libero_causal_contract import sample_options
from adapters.baseline_causal_runtime import source_hashes
p=argparse.ArgumentParser();p.add_argument("--method",choices=["act","dp","baku"],required=True)
p.add_argument("--data-only",action="store_true");args=p.parse_args()
job=os.environ["SLURM_JOB_ID"];out=ROOT/"artifacts/baseline-causal-check"/job
out.mkdir(parents=True,exist_ok=True)
start=time.time();report={"passed":False,"method":args.method,"training_updates":0,"data_only":args.data_only}
c=json.loads((ROOT/("configs/baselines_causal/"+args.method+"_seed0.json")).read_text())
split=json.loads((ROOT/"configs/act/split.json").read_text())
DATA=Path.home()/"scratch/robot_manipulation/datasets/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5"
folder=Path.home()/"scratch/robot_manipulation/baseline-causal-check"/job
def digest(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
 return h.hexdigest()
def cuda_batch(samples,keys):
 return {k:torch.from_numpy(np.stack([s[k] for s in samples])).cuda() for k in keys}
def write():
 (out/"result.json").write_text(json.dumps(report,indent=2)+"\n")
def cpu_state(model):return {k:v.detach().cpu() for k,v in model.state_dict().items()}
def act_check(f):
 from adapters.libero_act import LiberoACTDataset,normalize
 stats=json.loads((ROOT/"configs/act/normalization.json").read_text())
 assert set(stats["training_episodes"])==set(split["train"])
 ds=LiberoACTDataset(DATA,split["train"],stats,20,**sample_options(c))
 old=LiberoACTDataset(DATA,split["train"],stats,20)
 assert len(ds)==6156 and len(old)==6196
 offset=old_offset=count=0
 for name in split["train"]:
  d=f["data"][name];n=len(d["actions"])
  for t in sorted({0,1,7,n//2,n-2}):
   item=ds[offset+t];previous=old[old_offset+t]
   assert item["episode"]==name and item["timestep"]==t
   for k in ("images","state"):np.testing.assert_array_equal(item[k],previous[k])
   valid=min(20,n-t-1)
   np.testing.assert_array_equal(item["is_pad"],np.arange(20)>=valid)
   np.testing.assert_array_equal(item["actions"][:valid],normalize(d["actions"][t+1:t+1+valid].astype(np.float32),stats,"action"))
   assert np.all(item["actions"][valid:]==0)
   np.testing.assert_array_equal(previous["actions"][:min(20,n-t)],normalize(d["actions"][t:t+20].astype(np.float32),stats,"action"))
   count+=1
  offset+=n-1;old_offset+=n
 report["raw_data_windows_checked"]=count;report["eligible_training_anchors"]=len(ds)
 if args.data_only:return
 from adapters.act_model import build_act,act_loss
 model=build_act(chunk_size=20,pretrained_backbone=c["pretrained_backbone"]).cuda()
 b=cuda_batch([ds[0],ds[64]],("images","state","actions","is_pad"))
 captures=[]
 hook=model.encoder_action_proj.register_forward_pre_hook(lambda mod,inputs:captures.append(inputs[0].detach().clone()))
 opt=torch.optim.AdamW(model.parameters(),lr=c["lr"],weight_decay=1e-4)
 model.train();opt.zero_grad(set_to_none=True)
 try:pred,_,(mu,logvar)=model(b["state"],b["images"],None,b["actions"],b["is_pad"])
 finally:hook.remove()
 assert len(captures)==1;torch.testing.assert_close(captures[0],b["actions"],rtol=0,atol=0)
 loss=act_loss(pred,b["actions"],b["is_pad"],mu,logvar,c["kl_weight"])["loss"]
 assert torch.isfinite(loss);loss.backward()
 norm=torch.nn.utils.clip_grad_norm_(model.parameters(),c["grad_clip"]);assert torch.isfinite(norm) and norm>0
 opt.step();model.eval()
 with torch.no_grad():before=model(b["state"],b["images"],None)[0]
 assert before.shape==(2,20,7) and torch.isfinite(before).all()
 path=folder/"act.pt";torch.save(cpu_state(model),path)
 with torch.no_grad():next(model.parameters()).add_(.01)
 model.load_state_dict(torch.load(path,map_location="cpu"),strict=True)
 with torch.no_grad():after=model(b["state"],b["images"],None)[0]
 torch.testing.assert_close(before,after,rtol=1e-5,atol=1e-6)
 report.update(loss=float(loss.detach()),reload_max_abs=float((before-after).abs().max()),actual_encoder_target_verified=True)

def dp_check(f):
 from experiments.dp_run import CachedSamples
 from adapters.libero_sequence import LiberoSequenceDataset
 ds=CachedSamples(split["train"],**sample_options(c))
 ref=LiberoSequenceDataset(DATA,split["train"],16,2,**sample_options(c))
 old=LiberoSequenceDataset(DATA,split["train"],16,2)
 assert len(ds)==6156 and len(old)==6196
 offset=old_offset=count=0
 for name in split["train"]:
  d=f["data"][name];n=len(d["actions"]);raw=d["actions"][:].astype(np.float32)
  for t in sorted({0,1,7,n//2,n-2}):
   item=ds[offset+t];r=ref[offset+t];previous=old[old_offset+t]
   np.testing.assert_array_equal(item["action"],raw[np.clip(np.arange(t,t+16),0,n-1)])
   np.testing.assert_array_equal(item["action"][1:9],raw[np.clip(np.arange(t+1,t+9),0,n-1)])
   np.testing.assert_array_equal(previous["action"],raw[np.clip(np.arange(t-1,t+15),0,n-1)])
   np.testing.assert_array_equal(item["action"],r["action"])
   for k in item["obs"]:
    np.testing.assert_array_equal(item["obs"][k],r["obs"][k])
    np.testing.assert_array_equal(item["obs"][k],previous["obs"][k])
   count+=1
  offset+=n-1;old_offset+=n
 report["raw_data_windows_checked"]=count;report["eligible_training_anchors"]=len(ds)
 if args.data_only:return
 from adapters.dp_model import build_dp,make_normalizer,load_dp_state,assert_dp_placement
 policy=build_dp(c["pretrained_backbone"],c["inference_steps"])
 policy.set_normalizer(make_normalizer(DATA,split["train"]));policy.cuda()
 assert_dp_placement(policy,torch.device("cuda:0"))
 items=[ds[0],ds[64]]
 b={"obs":{k:torch.from_numpy(np.stack([s["obs"][k] for s in items])).cuda() for k in items[0]["obs"]},
    "action":torch.from_numpy(np.stack([s["action"] for s in items])).cuda()}
 opt=torch.optim.AdamW(policy.parameters(),lr=c["lr"],weight_decay=c["weight_decay"],betas=(.95,.999))
 captured=[];original=policy.noise_scheduler.add_noise
 def capture(original_samples,*a,**kw):
  captured.append(original_samples.detach().clone());return original(original_samples,*a,**kw)
 policy.noise_scheduler.add_noise=capture
 policy.train();opt.zero_grad(set_to_none=True)
 try:loss=policy.compute_loss(b)
 finally:policy.noise_scheduler.add_noise=original
 assert len(captured)==1
 torch.testing.assert_close(captured[0],policy.normalizer["action"].normalize(b["action"]),rtol=0,atol=0)
 assert torch.isfinite(loss);loss.backward()
 norm=torch.nn.utils.clip_grad_norm_(policy.parameters(),c["grad_clip"]);assert torch.isfinite(norm) and norm>0
 opt.step();policy.eval()
 rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all()
 with torch.no_grad():prediction=policy.predict_action(b["obs"])
 before=prediction["action"];assert before.shape==(2,8,7) and torch.isfinite(before).all()
 torch.testing.assert_close(before,prediction["action_pred"][:,1:9],rtol=0,atol=0)
 path=folder/"dp.pt";torch.save(cpu_state(policy),path)
 with torch.no_grad():next(policy.model.parameters()).add_(.01)
 load_dp_state(policy,torch.load(path,map_location="cpu"));assert_dp_placement(policy,torch.device("cuda:0"))
 torch.set_rng_state(rng);torch.cuda.set_rng_state_all(cuda_rng)
 with torch.no_grad():after=policy.predict_action(b["obs"])["action"]
 torch.testing.assert_close(before,after,rtol=1e-5,atol=1e-6)
 report.update(loss=float(loss.detach()),reload_max_abs=float((before-after).abs().max()),actual_diffusion_target_verified=True,output_slice=[1,9])

def baku_check(f):
 from experiments.baku_run import dataset,cpu,clone
 from agent.baku import BCAgent
 ds=dataset(c,"train")
 previous_config=dict(c);previous_config.pop("sample_contract");previous_config.pop("sample_anchor_policy")
 old=dataset(previous_config,"train")
 for subset in ("train","validation"):
  path=Path(c["data_root"])/subset/"libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket.pkl"
  assert digest(path)==c["pkl_sha256"][subset]
 assert len(ds._episodes[0])==len(split["train"])
 def at(data,episode,t):
  original_episode=data._sample_episode;original_randint=np.random.randint
  data._sample_episode=lambda env_idx=None:(data._episodes[0][episode],0)
  np.random.randint=lambda *a,**kw:t
  try:return data._sample()
  finally:data._sample_episode=original_episode;np.random.randint=original_randint
 count=eligible=0
 for ep,name in enumerate(split["train"]):
  d=f["data"][name];raw=d["actions"][:];n=len(raw);eligible+=n-1
  for t in sorted({0,1,7,n//2,n-2}):
   item=at(ds,ep,t);previous=at(old,ep,t)
   for k in ("pixels","pixels_egocentric","proprioceptive","task_emb"):
    np.testing.assert_array_equal(item[k],previous[k])
   expected=np.zeros((10,7));valid=min(10,n-t-1);expected[:valid]=raw[t+1:t+1+valid]
   np.testing.assert_allclose(item["actions"][0],expected/(1+1e-5),rtol=1e-7,atol=1e-7)
   raw_old=np.zeros((10,7));v=min(10,n-t);raw_old[:v]=raw[t:t+v]
   np.testing.assert_allclose(previous["actions"][0],raw_old/(1+1e-5),rtol=1e-7,atol=1e-7)
   count+=1
 assert eligible==6156
 report.update(raw_data_windows_checked=count,eligible_training_anchors=eligible,
               endpoint_padding="unchanged author zero command padding",original_pkls_verified=True)
 if args.data_only:return
 batch=next(iter(torch.utils.data.DataLoader(ds,batch_size=2,num_workers=0)))
 agent=BCAgent(**c["model_config"])
 actor_before=next(agent.actor.parameters()).detach().clone()
 metrics=agent.update(iter([clone(batch)]),step=0)
 assert metrics and all(np.isfinite(v) for v in metrics.values())
 for module in (agent.actor,agent.encoder,agent.proprio_projector,agent.language_projector):
  assert all(torch.isfinite(p).all() for p in module.parameters())
  grads=[p.grad for p in module.parameters() if p.grad is not None]
  assert grads and all(torch.isfinite(g).all() for g in grads)
 assert float((actor_before-next(agent.actor.parameters())).abs().max())>0
 obs={k:np.rint(batch[k][0,-1].numpy()*255).astype(np.uint8) for k in ("pixels","pixels_egocentric")}
 obs["proprioceptive"]=batch["proprioceptive"][0,-1].numpy()*(1+1e-5)
 prompt={"task_emb":batch["task_emb"][0].numpy()}
 def predict():
  agent.train(False);agent.buffer_reset()
  with torch.no_grad():return agent.act({k:v.copy() for k,v in obs.items()},prompt,ds.stats,0,1,eval_mode=True)
 before=predict();assert before.shape==(7,) and np.isfinite(before).all()
 snapshot={k:cpu(v) for k,v in agent.save_snapshot().items() if not k.endswith("_opt")}
 path=folder/"baku.pt";torch.save(snapshot,path)
 with torch.no_grad():next(agent.actor.parameters()).add_(.01)
 agent.load_snapshot(torch.load(path,map_location="cpu"),eval=True);after=predict()
 np.testing.assert_allclose(before,after,rtol=1e-5,atol=1e-6)
 report.update(loss=float(metrics["actor_loss"]),reload_max_abs=float(np.max(np.abs(before-after))))

try:
 torch.set_num_threads(2 if args.data_only else 4)
 if not args.data_only:assert torch.cuda.is_available()
 random.seed(0);np.random.seed(0);torch.manual_seed(0)
 assert digest(DATA)==c["dataset_sha256"]==split["dataset_sha256"]
 assert not set(split["train"])&set(split["validation"])
 folder.mkdir(parents=True,exist_ok=True)
 with h5py.File(DATA,"r") as f:
  {"act":act_check,"dp":dp_check,"baku":baku_check}[args.method](f)
 report["source_sha256"]=source_hashes(args.method)
 report["training_updates"]=0 if args.data_only else 1
 report["scope"]="data-only" if args.data_only else "real-data CUDA update, prediction and strict reload; not policy success"
 if not args.data_only:report["gpu"]=torch.cuda.get_device_name(0)
 report["passed"]=True
except Exception as exc:
 report["error"]=repr(exc);traceback.print_exc()
finally:
 report["wall_seconds"]=time.time()-start;write();print(json.dumps(report,indent=2),flush=True)
sys.exit(0 if report["passed"] else 1)
