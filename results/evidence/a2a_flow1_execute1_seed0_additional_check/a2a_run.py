"""LIBERO A2A pilot: author core; orchestration adapted from validated DP runner."""
import argparse, copy, hashlib, json, math, os, random, signal, sys, time
from collections import deque
from pathlib import Path
import h5py
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from adapters.a2a_model import build_a2a,make_normalizer,load_state,assert_placement
from adapters.libero_causal_contract import sample_options,sequence_indices
def build_model(pretrained,inference_steps):
    assert pretrained is False
    return build_a2a(sampling_steps=inference_steps)

DATA=Path.home()/"scratch/robot_manipulation/datasets/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5"
SOURCE="a5792ecf4e7f8fa4d85fe66ea9a50618138f925c"

class CachedSamples:
    """Same windows as LiberoSequenceDataset, raw uint8 images kept in CPU memory."""
    def __init__(self,episodes,action_offset=0,require_next_action=False):
        self.action_offset=action_offset;self.require_next_action=require_next_action
        self.episodes=list(episodes); self.data=[]
        with h5py.File(DATA,"r") as f:
            for name in episodes:
                d=f["data"][name]
                self.data.append({k:d[k][:] for k in (
                    "actions","obs/agentview_rgb","obs/eye_in_hand_rgb",
                    "obs/joint_states","obs/gripper_states")})
        lengths=[len(d["actions"]) for d in self.data]
        if not lengths or min(lengths)<=int(require_next_action):raise ValueError("No sample anchors")
        sequence_indices(lengths[0],0,16,8,action_offset,require_next_action)
        self.ends=np.cumsum([n-int(require_next_action) for n in lengths])
    def __len__(self): return int(self.ends[-1])
    def __getitem__(self,index):
        if not 0<=index<len(self):raise IndexError(index)
        ep=int(np.searchsorted(self.ends,index,side="right"))
        t=index-(int(self.ends[ep-1]) if ep else 0)
        d=self.data[ep]
        idx=sequence_indices(len(d["actions"]),t,16,8,self.action_offset,self.require_next_action)
        ids=idx["action_indices"];obs_ids=idx["obs_indices"]
        obs={}
        for key,source in (("agentview","obs/agentview_rgb"),("wrist","obs/eye_in_hand_rgb")):
            obs[key]=np.ascontiguousarray(d[source][obs_ids].transpose(0,3,1,2).astype(np.float32)/255)
        obs["agent_pos"]=np.concatenate([d["obs/joint_states"][obs_ids],d["obs/gripper_states"][obs_ids]],axis=-1).astype(np.float32)
        return {"obs":obs,"action":d["actions"][ids].astype(np.float32)}

def batch_from(ds,ids):
    s=[ds[int(i)] for i in ids]
    return {"obs":{k:torch.from_numpy(np.stack([x["obs"][k] for x in s])).cuda()
                   for k in ("agentview","wrist","agent_pos")},
            "action":torch.from_numpy(np.stack([x["action"] for x in s])).cuda()}

def write_json(path,value):
    path.write_text(json.dumps(value,indent=2)+"\n")

def cpu_state(model): return {k:v.detach().cpu() for k,v in model.state_dict().items()}

def train(c,out,job):
    from roboverse_learn.il.utils.ema_model import EMAModel
    from adapters.libero_sequence import LiberoSequenceDataset
    split=json.loads((ROOT/"configs/act/split.json").read_text())
    options=sample_options(c)
    ds=CachedSamples(split["train"],**options)
    ref=LiberoSequenceDataset(DATA,split["train"],16,8,**options)
    # Check optimized loader against the already GPU-validated loader at episode boundaries.
    for i in [0,len(ds)-1,int(ds.ends[0])-1,int(ds.ends[0])]:
        a,b=ds[i],ref[i]
        np.testing.assert_array_equal(a["action"],b["action"])
        for k in a["obs"]: np.testing.assert_array_equal(a["obs"][k],b["obs"][k])
    policy=build_model(c["pretrained_backbone"],c["inference_steps"])
    policy.set_normalizer(make_normalizer(DATA,split["train"]))
    policy.cuda(); assert_placement(policy,torch.device("cuda:0"))
    ema_policy=copy.deepcopy(policy)
    ema=EMAModel(ema_policy,power=c["ema_power"])
    opt=torch.optim.AdamW(policy.parameters(),lr=c["lr"],weight_decay=c["weight_decay"],betas=(0.95,0.999))
    def factor(step):
        if step<c["warmup_updates"]: return (step+1)/c["warmup_updates"]
        return 0.5*(1+math.cos(math.pi*(step-c["warmup_updates"])/max(1,c["updates"]-c["warmup_updates"])))
    scheduler=torch.optim.lr_scheduler.LambdaLR(opt,factor)
    folder=Path.home()/"scratch/robot_manipulation/a2a-training"/job
    folder.mkdir(parents=True,exist_ok=True)
    validation=CachedSamples(split["validation"],**options)
    val_ids=np.linspace(0,len(validation)-1,32,dtype=int)
    history=[]; start=time.time(); step=0;sample_order=hashlib.sha256()
    def save():
        target=folder/"latest.pt"; tmp=folder/"latest.tmp"
        torch.save({"policy":cpu_state(policy),"ema_policy":cpu_state(ema_policy),
                    "optimizer":opt.state_dict(),"lr_scheduler":scheduler.state_dict(),
                    "ema_step":ema.optimization_step,"config":c,"step":step,"split":split,
                    "source_commit":SOURCE,"history":history,"numpy_rng":np.random.get_state(),
                    "python_rng":random.getstate(),"torch_rng":torch.get_rng_state(),
                    "cuda_rng":torch.cuda.get_rng_state_all(),"sample_order_sha256":sample_order.hexdigest()},tmp)
        os.replace(tmp,target)
        return target
    for step in range(1,c["updates"]+1):
        policy.train()
        sample_ids=np.random.randint(len(ds),size=c["batch_size"])
        sample_order.update(sample_ids.astype("<i8").tobytes())
        batch=batch_from(ds,sample_ids)
        opt.zero_grad(set_to_none=True)
        loss=policy.compute_loss(batch)
        if not torch.isfinite(loss): raise RuntimeError("Nonfinite loss")
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(policy.parameters(),c["grad_clip"])
        if not torch.isfinite(norm): raise RuntimeError("Nonfinite gradient")
        opt.step(); scheduler.step(); ema.step(policy)
        if step==1 or step%50==0:
            row={"step":step,"loss":float(loss.detach()),"gradient_norm":float(norm),
                 "lr":scheduler.get_last_lr()[0],"elapsed_seconds":time.time()-start}
            history.append(row); print(json.dumps(row),flush=True)
            write_json(out/"metrics.json",history)
        if step%c["checkpoint_interval"]==0:
            ema_policy.eval()
            with torch.random.fork_rng(devices=[0]), torch.no_grad():
                torch.manual_seed(12345)
                val_losses=[]
                for offset in range(0,len(val_ids),c["batch_size"]):
                    val_losses.append(float(ema_policy.compute_loss(batch_from(validation,val_ids[offset:offset+c["batch_size"]]))))
            row={"step":step,"validation_total_loss":float(np.mean(val_losses)),"validation_samples":32}
            history.append(row); print(json.dumps(row),flush=True)
            write_json(out/"metrics.json",history)
            save()
        if time.time()-start>c["wall_budget_seconds"]:
            save()
            raise RuntimeError("Training wall budget exhausted; partial checkpoint retained")
    checkpoint=save()
    # Check EMA strict reload and dynamically reconstructed normalizer device.
    state=torch.load(checkpoint,map_location="cpu")
    load_state(ema_policy,state["ema_policy"])
    assert_placement(ema_policy,torch.device("cuda:0"))
    write_json(out/"result.json",{"completed":True,"updates":step,"checkpoint":str(checkpoint),
               "job_id":job,"gpu":torch.cuda.get_device_name(0),"config":c,
               "loader_parity_passed":True,"ema_reload_placement_passed":True,
               "sample_options":options,"training_samples":len(ds),"validation_samples":len(validation),
               "sample_order_sha256":sample_order.hexdigest(),
               "note":"Training completion only; success requires separate online evaluation."})

def live_obs(obs):
    return {"agentview":obs["agentview_image"].transpose(2,0,1).astype(np.float32)/255,
            "wrist":obs["robot0_eye_in_hand_image"].transpose(2,0,1).astype(np.float32)/255,
            "agent_pos":np.concatenate([obs["robot0_joint_pos"],obs["robot0_gripper_qpos"]]).astype(np.float32)}

def evaluate(c,out,checkpoint,job,protocol=None,resume_report=None):
    from libero.libero import benchmark,get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from PIL import Image
    protocol=protocol or {}
    allowed={"execution_horizon","checkpoint_sha256","purpose","sampling_steps"}
    if set(protocol)-allowed: raise ValueError("Unsupported evaluation overrides")
    sampling_steps=protocol.get("sampling_steps",c["inference_steps"])
    if sampling_steps not in (1,6):raise ValueError("Expected1or6flowsteps")
    execution_horizon=protocol.get("execution_horizon",c["execution_horizon"])
    if type(execution_horizon) is not int or not 1<=execution_horizon<=c["n_action_steps"]:
        raise ValueError("Invalid execution horizon")
    checksum=hashlib.sha256()
    with open(checkpoint,"rb") as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b""): checksum.update(chunk)
    checksum=checksum.hexdigest()
    if "checkpoint_sha256" in protocol:
        assert checksum==protocol["checkpoint_sha256"],"Frozen checkpoint checksum mismatch"
    saved=torch.load(checkpoint,map_location="cpu")
    assert saved["step"]==c["updates"] and saved["config"]==c
    assert saved["source_commit"]==SOURCE
    policy=build_model(False,sampling_steps)
    load_state(policy,saved["ema_policy"]); policy.cuda().eval()
    assert_placement(policy,torch.device("cuda:0"))
    del saved
    suite=benchmark.get_benchmark_dict()["libero_object"]()
    task=suite.get_task(0); states=suite.get_task_init_states(0)
    assert len(states)>max(c["rollout_initial_state_ids"])
    env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path("bddl_files"))/task.problem_folder/task.bddl_file),
                          camera_heights=128,camera_widths=128,render_gpu_device_id=0)
    report={"job_id":job,"completed":False,"checkpoint":str(checkpoint),"checkpoint_sha256":checksum,
            "checkpoint_training_step":c["updates"],"weights":"EMA","config":c,"evaluation_protocol":protocol,"execution_horizon":execution_horizon,"sampling_steps":sampling_steps,"rollouts":[],
            "gpu":torch.cuda.get_device_name(0),"task":task.name,"available_initial_states":len(states)}
    if resume_report:
        from adapters.eval_resume import restore_trials
        report["rollouts"]=restore_trials(resume_report,out,c,checksum,sampling_steps,execution_horizon)
        report["resume_parent_report"]=str(resume_report)
    finished={r["initial_state_id"] for r in report["rollouts"]}
    write_json(out/"result.json",report)
    try:
        for state_id,seed in zip(c["rollout_initial_state_ids"],c["rollout_seeds"]):
            if state_id in finished:continue
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            env.seed(seed); env.reset(); obs=env.set_init_state(states[state_id])
            for _ in range(c["settle_steps"]): obs,_,_,_=env.step(np.zeros(7))
            history=deque([live_obs(obs)]*8,maxlen=8)
            frames=[Image.fromarray(obs["agentview_image"][::-1])]; actions=[]; success=False; terminal=False; steps=0
            start=time.time()
            with torch.no_grad():
                while steps<c["rollout_horizon"] and not success and not terminal:
                    batch={k:torch.from_numpy(np.stack([h[k] for h in history])).unsqueeze(0).cuda() for k in history[0]}
                    predicted=policy.predict_action(batch)["action"][0].cpu().numpy()
                    assert predicted.shape==(8,7) and np.isfinite(predicted).all()
                    for action in predicted[:execution_horizon]:
                        executed=np.clip(action,-1,1)
                        obs,_,done,_=env.step(executed); steps+=1; terminal=bool(done)
                        if not np.isfinite(env.sim.data.qpos).all(): raise RuntimeError("Nonfinite simulation")
                        history.append(live_obs(obs)); actions.append(executed.tolist())
                        if steps%5==0: frames.append(Image.fromarray(obs["agentview_image"][::-1]))
                        success=bool(env.check_success())
                        if success or terminal or steps>=c["rollout_horizon"]: break
            frames[-1]=Image.fromarray(obs["agentview_image"][::-1])
            frames[0].save(out/("trial_%s.gif"%state_id),save_all=True,append_images=frames[1:],duration=250,loop=0)
            write_json(out/("trial_%s_actions.json"%state_id),actions)
            row={"initial_state_id":state_id,"seed":seed,"steps":steps,"success":success,"completed":True,"wall_seconds":time.time()-start}
            report["rollouts"].append(row); write_json(out/"result.json",report)
            print(json.dumps(row),flush=True)
        n=len(report["rollouts"]); s=sum(r["success"] for r in report["rollouts"])
        assert n==len(c["rollout_initial_state_ids"])
        p=s/n; z=1.959963984540054; den=1+z*z/n
        center=(p+z*z/(2*n))/den; half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
        report.update(completed=True,successes=s,completed_rollouts=n,success_rate=p,wilson95=[center-half,center+half])
    finally:
        write_json(out/"result.json",report); env.close()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=["train","eval"]); parser.add_argument("--training-job")
    parser.add_argument("--config",default="configs/a2a/train_seed0.json")
    parser.add_argument("--eval-protocol")
    parser.add_argument("--checkpoint")
    parser.add_argument("--resume-eval-report")
    parser.add_argument("--alignment-gate-job")
    args=parser.parse_args()
    assert os.environ.get("SLURM_JOB_ID"),"Run on Slurm compute nodes"
    torch.set_num_threads(4)
    c=json.loads((ROOT/args.config).read_text())
    sample_options(c)  # Reject unknown contracts before training or evaluation.
    random.seed(c["seed"]); np.random.seed(c["seed"]); torch.manual_seed(c["seed"])
    assert torch.cuda.is_available()
    actual=hashlib.sha256(DATA.read_bytes()).hexdigest()
    assert actual==c["dataset_sha256"],"Dataset checksum mismatch"
    job=os.environ["SLURM_JOB_ID"]
    out=ROOT/"artifacts"/("a2a-train" if args.mode=="train" else "a2a-eval")/job
    out.mkdir(parents=True,exist_ok=True); write_json(out/"config.json",c)
    (out/"a2a_run.py").write_text(Path(__file__).read_text())
    (out/"a2a_model.py").write_text((ROOT/"adapters/a2a_model.py").read_text())
    (out/"a2a_policy.py").write_text((ROOT/"third_party/A2A/roboverse_learn/il/policies/a2a/a2a_policy.py").read_text())
    (out/"author_policy_config.yaml").write_text((ROOT/c["policy_config"]).read_text())
    for source in ("adapters/libero_sequence.py","adapters/libero_causal_contract.py"):
        (out/Path(source).name).write_text((ROOT/source).read_text())
    if args.mode=="train":
        if args.alignment_gate_job:
            alignment=json.loads((ROOT/"artifacts/a2a-causal-integration"/args.alignment_gate_job/"result.json").read_text())
            assert alignment["passed"],"Causal integration gate failed"
            write_json(out/"alignment_gate.json",{"job_id":args.alignment_gate_job,"report":alignment})
            assert c["sample_contract"] in alignment["verified_contracts"]
            for source,expected in alignment["source_sha256"].items():
                assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest()==expected,source+" changed after gate"
        gate=json.loads((ROOT/"artifacts/a2a-integration"/c["integration_job"]/"result.json").read_text())
        assert gate["passed"] and gate["training_updates"]==1,"Integration gate failed"
        train(c,out,job)
    else:
        assert args.training_job or args.checkpoint
        checkpoint=Path(args.checkpoint).expanduser() if args.checkpoint else Path(json.loads((ROOT/"artifacts/a2a-train"/args.training_job/"result.json").read_text())["checkpoint"])
        protocol=json.loads((ROOT/args.eval_protocol).read_text()) if args.eval_protocol else None
        evaluate(c,out,checkpoint,job,protocol,args.resume_eval_report)

if __name__=="__main__": main()
