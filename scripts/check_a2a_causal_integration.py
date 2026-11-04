"""Paired-loader/actual author loss slice/update/reload gate for timing-only experiment."""
import hashlib,json,os,random,sys,time,traceback
from pathlib import Path
import h5py,numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from adapters.libero_causal_contract import CONTRACT_VERSION,LEGACY_CONTRACT,sample_options,causal_indices
from adapters.libero_sequence import LiberoSequenceDataset
from experiments.a2a_run import CachedSamples,DATA
from adapters.a2a_model import build_a2a,make_normalizer,load_state,assert_placement
data_only="--data-only" in sys.argv
job=os.environ["SLURM_JOB_ID"];out=ROOT/("artifacts/a2a-causal-data" if data_only else "artifacts/a2a-causal-integration")/job
out.mkdir(parents=True,exist_ok=True)
report={"passed":False,"training_updates":0,"verified_contracts":[]};start=time.time()
def write(): (out/"result.json").write_text(json.dumps(report,indent=2)+"\n")
def batch(samples):
 return {"obs":{k:torch.from_numpy(np.stack([s["obs"][k] for s in samples])).cuda() for k in samples[0]["obs"]},
         "action":torch.from_numpy(np.stack([s["action"] for s in samples])).cuda()}
def digest(model):
 h=hashlib.sha256()
 for name,p in model.state_dict().items():
  h.update(name.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
try:
 if not data_only:assert torch.cuda.is_available()
 torch.set_num_threads(2 if data_only else 4)
 split=json.loads((ROOT/"configs/act/split.json").read_text())
 h=hashlib.sha256()
 with DATA.open("rb") as f:
  for block in iter(lambda:f.read(8*1024*1024),b""):h.update(block)
 assert h.hexdigest()==split["dataset_sha256"]
 report["dataset_sha256"]=h.hexdigest()
 configs=[json.loads((ROOT/("configs/a2a/train_timing_"+v+"_seed0_10000.json")).read_text()) for v in ("control","causal")]
 differing={k for k in set(configs[0])|set(configs[1]) if configs[0].get(k)!=configs[1].get(k)}
 assert differing=={"purpose","sample_contract"},differing
 opts=[sample_options(c) for c in configs]
 datasets=[CachedSamples(split["train"],**o) for o in opts]
 assert len(datasets[0])==len(datasets[1])==6156
 references=[LiberoSequenceDataset(DATA,split["train"],16,8,**o) for o in opts]
 count=0
 with h5py.File(DATA,"r") as f:
  start_index=0
  for ep,name in enumerate(split["train"]):
   n=len(f["data"][name]["actions"]);a=f["data"][name]["actions"][:]
   for t in sorted({0,1,7,n//2,n-2}):
    i=start_index+t
    pair=[ds[i] for ds in datasets]
    for k in pair[0]["obs"]:np.testing.assert_array_equal(pair[0]["obs"][k],pair[1]["obs"][k])
    hist=np.clip(np.arange(t-7,t+1),0,n-1)
    raw_q=np.concatenate([f["data"][name]["obs/joint_states"][:],f["data"][name]["obs/gripper_states"][:]],axis=-1)
    np.testing.assert_array_equal(pair[1]["obs"]["agent_pos"],raw_q[hist].astype(np.float32))
    for variant in (0,1):
     expected=a[np.clip(np.arange(t+variant,t+variant+8),0,n-1)].astype(np.float32)
     np.testing.assert_array_equal(pair[variant]["action"][7:15],expected)
     ref=references[variant][i]
     assert ref["episode"]==name and ref["timestep"]==t
     np.testing.assert_array_equal(pair[variant]["action"],ref["action"])
     for k in pair[variant]["obs"]:np.testing.assert_array_equal(pair[variant]["obs"][k],ref["obs"][k])
    causal=causal_indices(n,t)
    np.testing.assert_array_equal(pair[1]["action"][7:15],a[causal["future_indices"]].astype(np.float32))
    count+=1
   start_index+=n-1
 # Legacy defaults retain all anchors, with same-index windows.
 old=LiberoSequenceDataset(DATA,[split["train"][0]],16,8)
 with h5py.File(DATA,"r") as f:
  d=f["data"][split["train"][0]];n=len(d["actions"])
  assert len(old)==n
  for t in (0,7,n-1):
   np.testing.assert_array_equal(old[t]["action"],d["actions"][:][np.clip(np.arange(t-7,t+9),0,n-1)].astype(np.float32))
 report["paired_windows_checked"]=count;report["paired_training_anchors"]=len(datasets[0])
 with h5py.File(DATA,"r") as f:
  report["paired_validation_anchors"]=sum(len(f["data"][n]["actions"])-1 for n in split["validation"])
 report["legacy_default_compatibility_passed"]=True
 print("Paired raw-HDF5 target checks and loader parity passed",flush=True);write()
 if data_only:
  report.update(passed=True,scope="data_only; GPU loss/update/reload gate remains separate")
  sys.exit(0)
 report["variants"]=[]
 init_hashes=[]
 for variant,ds in enumerate(datasets):
  random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
  policy=build_a2a(6);policy.set_normalizer(make_normalizer(DATA,split["train"]));policy.cuda()
  assert_placement(policy,torch.device("cuda:0"));init_hashes.append(digest(policy))
  b=batch([ds[0],ds[64]])
  captured=[]
  handle=policy.action_encoder.register_forward_pre_hook(lambda module,args: captured.append(args[0].detach().clone()))
  optimizer=torch.optim.AdamW(policy.parameters(),lr=1e-4,weight_decay=1e-6,betas=(.95,.999))
  policy.train();optimizer.zero_grad(set_to_none=True)
  try:loss=policy.compute_loss(b)
  finally:handle.remove()
  expected=policy.normalizer["action"].normalize(b["action"])[:,7:15]
  assert len(captured)==1;torch.testing.assert_close(captured[0],expected,rtol=0,atol=0)
  assert torch.isfinite(loss);loss.backward()
  norm=torch.nn.utils.clip_grad_norm_(policy.parameters(),1.)
  assert torch.isfinite(norm) and norm>0
  for label,module in (("history",policy.history_action_encoder),("future",policy.action_encoder),("decoder",policy.action_decoder),("flow",policy.flow_net)):
   grads=[p.grad for p in module.parameters() if p.grad is not None]
   assert grads and all(torch.isfinite(g).all() for g in grads),label
  optimizer.step();report["training_updates"]+=1
  policy.eval()
  with torch.no_grad():pred=policy.predict_action(b["obs"])["action"]
  assert pred.shape==(2,8,7) and torch.isfinite(pred).all()
  folder=Path.home()/"scratch/robot_manipulation/a2a-causal-integration"/job;folder.mkdir(parents=True,exist_ok=True)
  target=folder/("variant_"+str(variant)+".pt")
  torch.save({"policy":{k:v.detach().cpu() for k,v in policy.state_dict().items()}},target)
  with torch.no_grad():next(policy.flow_net.parameters()).add_(.01)
  load_state(policy,torch.load(target,map_location="cpu")["policy"])
  with torch.no_grad():reloaded=policy.predict_action(b["obs"])["action"]
  torch.testing.assert_close(pred,reloaded,rtol=1e-5,atol=1e-6)
  report["variants"].append({"contract":configs[variant]["sample_contract"],"loss":float(loss.detach()),
    "gradient_norm":float(norm),"actual_future_encoder_input_matches":True,
    "reload_action_max_abs":float((pred-reloaded).abs().max()),"output_shape":list(pred.shape)})
  report["verified_contracts"].append(configs[variant]["sample_contract"])
  del optimizer,policy,b,pred,reloaded,captured
  torch.cuda.empty_cache()
 assert init_hashes[0]==init_hashes[1],"Paired initialization differs"
 report["initial_state_sha256"]=init_hashes[0]
 report["source_sha256"]={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in (
   "adapters/libero_causal_contract.py","adapters/libero_sequence.py","adapters/a2a_model.py","experiments/a2a_run.py",
   "third_party/A2A/roboverse_learn/il/policies/a2a/a2a_policy.py")}
 report["gpu"]=torch.cuda.get_device_name(0)
 report["passed"]=True
except Exception as exc:
 report["error"]=repr(exc);traceback.print_exc()
finally:
 report["wall_seconds"]=time.time()-start;write();print(json.dumps(report,indent=2),flush=True)
sys.exit(0 if report["passed"] else 1)
