"""Single-task BAKU pilot using author update/act; no suite reproduction claim."""
import argparse,hashlib,json,math,os,random,sys,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"third_party/BAKU/baku"))
from agent.baku import BCAgent
from read_data.libero import BCDataset
from adapters.baku_live import online_observation,identity_stats
SOURCE="47e367059191de30ac4b6353ff5615fdef11d8c5"
TASK="pick_up_the_alphabet_soup_and_place_it_in_the_basket"
def write(path,x):path.write_text(json.dumps(x,indent=2)+"\n")
def cpu(x):
    if torch.is_tensor(x):return x.detach().cpu()
    if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
    if isinstance(x,list):return [cpu(v) for v in x]
    if isinstance(x,tuple):return tuple(cpu(v) for v in x)
    return x
def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""):h.update(block)
    return h.hexdigest()
def dataset(c,subset):
    from adapters.libero_causal_contract import sample_options
    options=sample_options(c)
    cls=BCDataset
    if options["action_offset"]:
        from adapters.baku_causal import CausalBCDataset
        cls=CausalBCDataset
    return cls(path=str(Path(c["data_root"])/subset),suite="libero_object",scenes=["object"],
        tasks=[{"object":[TASK]}],num_demos_per_task=40 if subset=="train" else 10,
        obs_type="pixels",history=False,history_len=10,prompt="text",
        temporal_agg=True,num_queries=10,img_size=128)
def clone(batch):return {k:v.clone() for k,v in batch.items()}
def train(c,out,job):
    agent=BCAgent(**c["model_config"])
    ds=dataset(c,"train")
    stream=iter(torch.utils.data.DataLoader(ds,batch_size=c["batch_size"],num_workers=0))
    val=dataset(c,"validation")
    py_rng=random.getstate();np_rng=np.random.get_state()
    with torch.random.fork_rng(devices=[0]):
        random.seed(12345);np.random.seed(12345);torch.manual_seed(12345)
        valstream=iter(torch.utils.data.DataLoader(val,batch_size=16,num_workers=0))
        valbatches=[next(valstream),next(valstream)]
    random.setstate(py_rng);np.random.set_state(np_rng)
    folder=Path.home()/"scratch/robot_manipulation/baku-training"/job;folder.mkdir(parents=True,exist_ok=True)
    history=[];step=0;start=time.time()
    def validate():
        agent.train(False);losses=[]
        with torch.no_grad():
            for b in valbatches:
                losses.append(agent.update(iter([clone(b)]),step,update=False)["actor_loss"])
        agent.train(True)
        row={"step":step,"validation_actor_loss":float(np.mean(losses)),"validation_samples":32}
        history.append(row);print(json.dumps(row),flush=True)
    def save():
        snapshot=agent.save_snapshot()
        model={k:cpu(v) for k,v in snapshot.items() if not k.endswith("_opt")}
        opts={k:cpu(v.state_dict()) for k,v in snapshot.items() if k.endswith("_opt")}
        path=folder/"latest.pt";tmp=folder/"latest.tmp"
        torch.save({"snapshot":model,"optimizer_states":opts,"config":c,"step":step,
                    "source_commit":SOURCE,"history":history,"python_rng":random.getstate(),
                    "numpy_rng":np.random.get_state(),"torch_rng":torch.get_rng_state(),
                    "cuda_rng":torch.cuda.get_rng_state_all()},tmp)
        os.replace(tmp,path);return path
    validate()
    for step in range(1,c["updates"]+1):
        metrics=agent.update(stream,step-1)
        assert np.isfinite(metrics["actor_loss"]),metrics
        if step==1 or step%50==0:
            row={"step":step,"actor_loss":metrics["actor_loss"],"elapsed_seconds":time.time()-start}
            history.append(row);print(json.dumps(row),flush=True)
            write(out/"metrics.json",history)
        if step%c["checkpoint_interval"]==0:
            validate();write(out/"metrics.json",history);save()
        if time.time()-start>c["wall_budget_seconds"]:
            save();raise RuntimeError("Wall budget exhausted;partialcheckpoint retained")
    checkpoint=save()
    # All model modules and optimizer states stored separately; exact-resume runner not implemented.
    saved=torch.load(checkpoint,map_location="cpu")
    assert saved["step"]==c["updates"]
    agent.load_snapshot(saved["snapshot"],eval=True)
    write(out/"result.json",{"completed":True,"updates":step,"checkpoint":str(checkpoint),
          "config":c,"gpu":torch.cuda.get_device_name(0),"job_id":job,
          "note":"Training completion only; online success requires dependent evaluation."})
def evaluate(c,out,checkpoint,job,protocol=None):
    from libero.libero import benchmark,get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    protocol=protocol or {}
    if set(protocol)-{"initial_state_ids","rollout_seeds","checkpoint_sha256","purpose"}:
        raise ValueError("Unsupported evaluation override")
    ids=protocol.get("initial_state_ids",c["initial_state_ids"])
    seeds=protocol.get("rollout_seeds",c["rollout_seeds"])
    assert len(ids)==len(seeds)>0 and len(set(ids))==len(ids)
    assert all(type(i) is int and i>=0 for i in ids)
    if "checkpoint_sha256" in protocol:
        assert digest(checkpoint)==protocol["checkpoint_sha256"],"Frozen checkpoint changed"
    saved=torch.load(checkpoint,map_location="cpu")
    assert saved["step"]==c["updates"] and saved["config"]==c and saved["source_commit"]==SOURCE
    agent=BCAgent(**c["model_config"]);agent.load_snapshot(saved["snapshot"],eval=True);del saved
    prompt={"task_emb":np.load(Path(c["data_root"])/"task_embedding.npy")}
    suite=benchmark.get_benchmark_dict()["libero_object"]();task=suite.get_task(0)
    assert task.name==TASK
    states=suite.get_task_init_states(0);assert len(states)>max(ids)
    env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path("bddl_files"))/task.problem_folder/task.bddl_file),
                          camera_heights=128,camera_widths=128,render_gpu_device_id=0)
    report={"completed":False,"job_id":job,"checkpoint":str(checkpoint),"checkpoint_sha256":digest(checkpoint),
            "checkpoint_training_step":c["updates"],"config":c,"evaluation_protocol":protocol,"rollouts":[],
            "training_updates_this_run":0,"gpu":torch.cuda.get_device_name(0),"available_initial_states":len(states)}
    write(out/"result.json",report)
    try:
        for initial_id,seed in zip(ids,seeds):
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            env.seed(seed);env.reset();obs=env.set_init_state(states[initial_id])
            for _ in range(c["settle_steps"]):obs,_,_,_=env.step(np.zeros(7))
            agent.buffer_reset();frames=[Image.fromarray(obs["agentview_image"][::-1])]
            actions=[];success=False;start=time.time()
            with torch.no_grad():
                for step in range(c["rollout_horizon"]):
                    action=np.asarray(agent.act(online_observation(obs),prompt,identity_stats(),step,step,eval_mode=True))
                    assert action.shape==(7,) and np.isfinite(action).all()
                    executed=np.clip(action,-1,1);obs,reward,done,_=env.step(executed)
                    assert np.isfinite(env.sim.data.qpos).all()
                    success=bool(env.check_success())
                    actions.append({"step":step,"action":executed.tolist(),"reward":float(reward),"done":bool(done)})
                    if (step+1)%5==0:frames.append(Image.fromarray(obs["agentview_image"][::-1]))
                    if success or done:break
            frames.append(Image.fromarray(obs["agentview_image"][::-1]))
            frames[0].save(out/("trial_%d.gif"%initial_id),save_all=True,append_images=frames[1:],duration=250,loop=0)
            write(out/("trial_%d_actions.json"%initial_id),actions)
            row={"initial_state_id":initial_id,"seed":seed,"steps":len(actions),"success":success,
                 "completed":True,"wall_seconds":time.time()-start}
            report["rollouts"].append(row);write(out/"result.json",report);print(json.dumps(row),flush=True)
        n=len(report["rollouts"]);assert n==len(ids)
        successes=sum(r["success"] for r in report["rollouts"]);p=successes/n;z=1.959963984540054
        den=1+z*z/n;center=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
        report.update(completed=True,successes=successes,completed_rollouts=n,success_rate=p,
                      wilson95=[center-half,center+half])
    finally:
        write(out/"result.json",report);env.close()
def main():
    parser=argparse.ArgumentParser();parser.add_argument("mode",choices=["train","eval"])
    parser.add_argument("--training-job");parser.add_argument("--config",default="configs/baku/train_seed0.json")
    parser.add_argument("--checkpoint");parser.add_argument("--eval-protocol");args=parser.parse_args()
    assert os.environ.get("SLURM_JOB_ID")
    c=json.loads((ROOT/args.config).read_text())
    random.seed(c["seed"]);np.random.seed(c["seed"]);torch.manual_seed(c["seed"]);torch.set_num_threads(4)
    assert torch.cuda.is_available()
    data=Path.home()/"scratch/robot_manipulation/datasets/libero_object"/(TASK+"_demo.hdf5")
    assert digest(data)==c["dataset_sha256"]
    for subset in ("train","validation"):
        path=Path(c["data_root"])/subset/"libero_object"/(TASK+".pkl")
        assert digest(path)==c["pkl_sha256"][subset],"Data export changed"
    job=os.environ["SLURM_JOB_ID"];out=ROOT/"artifacts"/("baku-train" if args.mode=="train" else "baku-eval")/job
    out.mkdir(parents=True,exist_ok=True);write(out/"config.json",c)
    (out/"baku_run.py").write_text(Path(__file__).read_text())
    from adapters.baseline_causal_runtime import record_run
    record_run("baku",out,c,training=args.mode=="train")
    if args.mode=="train":train(c,out,job)
    else:
        assert args.training_job or args.checkpoint
        checkpoint=Path(args.checkpoint).expanduser() if args.checkpoint else Path.home()/"scratch/robot_manipulation/baku-training"/args.training_job/"latest.pt"
        protocol=json.loads((ROOT/args.eval_protocol).read_text()) if args.eval_protocol else None
        evaluate(c,out,checkpoint,job,protocol)
if __name__=="__main__":main()
