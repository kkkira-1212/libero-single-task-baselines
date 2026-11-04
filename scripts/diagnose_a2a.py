"""Frozen A2A diagnosis: raw/EMA, direct action AE vs flow, latent statistics."""
import json,os,sys,time
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from adapters.a2a_model import build_a2a,load_state,assert_placement
from experiments.a2a_run import CachedSamples,batch_from
job=os.environ["SLURM_JOB_ID"];out=ROOT/"artifacts/a2a-diagnosis"/job;out.mkdir(parents=True,exist_ok=True)
r={"completed":False,"training_updates":0,"purpose":"frozen_A2A_raw_EMA_AE_flow_diagnosis","results":{}}
start=time.time()
try:
 torch.set_num_threads(4);np.random.seed(0);torch.manual_seed(0)
 train_report=json.loads((ROOT/"artifacts/a2a-train/14447500/result.json").read_text())
 assert train_report["completed"] and train_report["updates"]==10000
 checkpoint=Path(train_report["checkpoint"])
 assert checkpoint.is_file(),"Training report checkpoint missing"
 r["checkpoint"]=str(checkpoint)
 eval_report=json.loads((ROOT/"artifacts/a2a-eval/14447501/result.json").read_text())
 import hashlib
 checksum=hashlib.sha256()
 with checkpoint.open("rb") as f:
  for block in iter(lambda:f.read(8*1024*1024),b""):checksum.update(block)
 assert checksum.hexdigest()==eval_report["checkpoint_sha256"],"Checkpoint changed since evaluation"
 r["checkpoint_sha256"]=checksum.hexdigest()
 saved=torch.load(checkpoint,map_location="cpu")
 assert saved["step"]==10000 and saved["source_commit"]=="a5792ecf4e7f8fa4d85fe66ea9a50618138f925c"
 policy=build_a2a(6).cuda();split=saved["split"]
 for weight_key in ("policy","ema_policy"):
  load_state(policy,saved[weight_key]);policy.eval();assert_placement(policy,torch.device("cuda:0"))
  result={}
  for subset,key in (("train","train"),("validation","validation")):
   ds=CachedSamples(split[key]);ids=np.linspace(0,len(ds)-1,32,dtype=int)
   target_list=[];ae_list=[];predictions={1:[],6:[]};history_latents=[];target_latents=[]
   with torch.no_grad():
    for start_id in range(0,len(ids),4):
     batch=batch_from(ds,ids[start_id:start_id+4])
     target=batch["action"][:,7:15]
     normalized=policy.normalizer["action"].normalize(target)
     latent=policy.action_encoder(normalized)
     direct=policy.normalizer["action"].unnormalize(policy.action_decoder(latent))
     target_list.append(target.cpu().numpy());ae_list.append(direct.cpu().numpy())
     state=policy.normalizer["agent_pos"].normalize(batch["obs"]["agent_pos"])
     history_latents.append(policy.history_action_encoder(state).cpu().numpy())
     target_latents.append(latent.cpu().numpy())
     for steps in (1,6):
      policy.num_sampling_steps=steps
      predicted=policy.predict_action(batch["obs"])["action"]
      assert torch.isfinite(predicted).all()
      predictions[steps].append(predicted.cpu().numpy())
   target=np.concatenate(target_list);ae=np.concatenate(ae_list)
   def errors(pred):
    diff=pred-target
    return {"mse_per_dimension":np.mean(diff**2,axis=(0,1)).tolist(),
            "first_action_rmse":float(np.sqrt(np.mean(diff[:,0]**2))),
            "gripper_sign_accuracy":float(np.mean(np.sign(pred[:,:,6])==np.sign(target[:,:,6]))),
            "predicted_per_dimension_std":np.std(pred,axis=(0,1)).tolist(),
            "target_per_dimension_std":np.std(target,axis=(0,1)).tolist()}
   hl=np.concatenate(history_latents);tl=np.concatenate(target_latents)
   result[subset]={"samples":len(ids),"action_AE_reconstruction":errors(ae),
       "flow_1":errors(np.concatenate(predictions[1])),"flow_6":errors(np.concatenate(predictions[6])),
       "history_latent_mean_norm":float(np.mean(np.linalg.norm(hl,axis=-1))),
       "future_latent_mean_norm":float(np.mean(np.linalg.norm(tl,axis=-1))),
       "history_latent_mean_std":float(np.std(hl,axis=0).mean()),
       "future_latent_mean_std":float(np.std(tl,axis=0).mean()),
       "note":"32uniformlyselectedtimestepwindows,notfullsplitbenchmark"}
   del ds
  r["results"][weight_key]=result
  (out/"result.json").write_text(json.dumps(r,indent=2)+"\n")
 r["completed"]=True
except Exception as exc:
 r["error"]=repr(exc)
 import traceback;traceback.print_exc()
finally:
 r["wall_seconds"]=time.time()-start
 (out/"result.json").write_text(json.dumps(r,indent=2)+"\n")
 print(json.dumps(r,indent=2),flush=True)
sys.exit(0 if r["completed"] else 1)
