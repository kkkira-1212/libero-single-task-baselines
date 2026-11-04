"""Frozen A2A confirmation evaluation; same model/evaluation loop as validated runner.

Derived from experiments/a2a_run.py evaluate; only adds explicit rollout-ID/seed
overrides while preserving exact checkpoint training-config validation.
No training path is modified. No partial-report resume is exposed here.
"""
import argparse,hashlib,json,math,os,random,sys,time
from collections import deque
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.a2a_run import SOURCE,build_model,live_obs,write_json
from adapters.a2a_model import load_state,assert_placement

def validate_rollouts(protocol,config,available):
    ids=protocol.get("initial_state_ids",config["rollout_initial_state_ids"])
    seeds=protocol.get("rollout_seeds",config["rollout_seeds"])
    if not isinstance(ids,list) or not ids or any(type(i) is not int or not 0<=i<available for i in ids):
        raise ValueError("Invalid initial-state IDs")
    if len(set(ids))!=len(ids):
        raise ValueError("Repeated initial-state IDs")
    if not isinstance(seeds,list) or len(seeds)!=len(ids) or any(type(i) is not int or not 0<=i<2**32 for i in seeds):
        raise ValueError("Invalid rollout seeds")
    return ids,seeds

def evaluate_confirm(c,out,checkpoint,job,protocol=None):
    from libero.libero import benchmark,get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from PIL import Image
    protocol=protocol or {}
    allowed={"execution_horizon","checkpoint_sha256","purpose","sampling_steps","initial_state_ids","rollout_seeds","settle_gripper_command"}
    if set(protocol)-allowed: raise ValueError("Unsupported evaluation overrides")
    settle_gripper=protocol.get("settle_gripper_command",0.0)
    if type(settle_gripper) not in (int,float) or not np.isfinite(settle_gripper) or not -1<=settle_gripper<=1:
        raise ValueError("Invalid diagnostic settle gripper command")
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
    ids,seeds=validate_rollouts(protocol,c,len(states))
    env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path("bddl_files"))/task.problem_folder/task.bddl_file),
                          camera_heights=128,camera_widths=128,render_gpu_device_id=0)
    report={"job_id":job,"completed":False,"checkpoint":str(checkpoint),"checkpoint_sha256":checksum,
            "checkpoint_training_step":c["updates"],"weights":"EMA","config":c,"evaluation_protocol":protocol,"execution_horizon":execution_horizon,"sampling_steps":sampling_steps,"rollouts":[],
            "gpu":torch.cuda.get_device_name(0),"task":task.name,"available_initial_states":len(states)}
    report["rollout_initial_state_ids"]=ids
    report["rollout_seeds"]=seeds
    write_json(out/"result.json",report)
    try:
        for state_id,seed in zip(ids,seeds):
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            env.seed(seed); env.reset(); obs=env.set_init_state(states[state_id])
            settle_action=np.zeros(7);settle_action[-1]=settle_gripper
            for _ in range(c["settle_steps"]): obs,_,_,_=env.step(settle_action)
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
        assert n==len(ids)
        p=s/n; z=1.959963984540054; den=1+z*z/n
        center=(p+z*z/(2*n))/den; half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
        report.update(completed=True,successes=s,completed_rollouts=n,success_rate=p,wilson95=[center-half,center+half])
    finally:
        write_json(out/"result.json",report); env.close()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--frozen-manifest")
    p.add_argument("--training-job")
    p.add_argument("--config")
    p.add_argument("--protocol",required=True)
    args=p.parse_args()
    assert os.environ.get("SLURM_JOB_ID"),"Use a Slurm compute allocation"
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    protocol=json.loads((ROOT/args.protocol).read_text())
    if args.frozen_manifest:
        assert not args.training_job and not args.config
        frozen=json.loads((ROOT/args.frozen_manifest).read_text())
        config=frozen["config"];checkpoint=Path(frozen["checkpoint"])
        if "checkpoint_sha256" in protocol:
            assert protocol["checkpoint_sha256"]==frozen["checkpoint_sha256"]
        protocol["checkpoint_sha256"]=frozen["checkpoint_sha256"]
    else:
        assert args.training_job and args.config
        config=json.loads((ROOT/args.config).read_text())
        trained=json.loads((ROOT/"artifacts/a2a-train"/args.training_job/"result.json").read_text())
        assert trained["completed"] and trained["config"]==config
        checkpoint=Path(trained["checkpoint"])
    job=os.environ["SLURM_JOB_ID"];out=ROOT/"artifacts/a2a-eval"/job
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/"confirmation_inputs.json",{"config":config,"protocol":protocol,"checkpoint":str(checkpoint)})
    for path in ("experiments/a2a_confirm_eval.py","experiments/a2a_run.py","adapters/a2a_model.py",
                 "adapters/libero_causal_contract.py"):
        (out/Path(path).name).write_text((ROOT/path).read_text())
    random.seed(config["seed"]);np.random.seed(config["seed"]);torch.manual_seed(config["seed"])
    evaluate_confirm(config,out,checkpoint,job,protocol)

if __name__=="__main__":main()
