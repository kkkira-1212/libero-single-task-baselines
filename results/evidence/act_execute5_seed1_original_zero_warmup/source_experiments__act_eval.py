"""Policy-only diagnostic evaluation; no optimizer construction or training."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
stop_requested=False
def stop(signum,frame):
    global stop_requested
    stop_requested=True

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--checkpoint",required=True)
    parser.add_argument("--minimum-training-step",type=int,default=0)
    parser.add_argument("--protocol",default=None)
    args=parser.parse_args()
    assert os.environ.get("SLURM_JOB_ID")
    signal.signal(signal.SIGUSR1,stop)
    signal.signal(signal.SIGTERM,stop)
    job=os.environ["SLURM_JOB_ID"]
    out=ROOT/"artifacts/act-eval"/job
    out.mkdir(parents=True,exist_ok=True)
    start=time.time()
    report={"job_id":job,"host":socket.gethostname(),"execution_ok":False,
            "training_updates_this_run":0,"rollouts":[],
            "purpose":"two_rollout_pipeline_diagnostic_not_benchmark",
            "checkpoint":args.checkpoint,"timings":{}}
    env=None
    try:
        import numpy as np
        import torch
        import imageio
        from adapters.act_model import build_act
        from adapters.libero_act import online_observation,decode_action
        assert torch.cuda.is_available()
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark=False
        report.update(gpu=torch.cuda.get_device_name(0),torch=torch.__version__)
        protocol=None
        if args.protocol:
            import hashlib
            protocol=json.loads(Path(args.protocol).read_text())
            assert str(Path(args.checkpoint).resolve())==str(Path(protocol["checkpoint"]).resolve())
            checksum=hashlib.sha256()
            with open(args.checkpoint,"rb") as stream:
                for part in iter(lambda:stream.read(4*1024*1024),b""):
                    checksum.update(part)
            assert checksum.hexdigest()==protocol["checkpoint_sha256"], "Frozen checkpoint changed"
            assert not set(protocol["rollout_initial_state_ids"]) & set(protocol["previous_development_ids"])
            assert len(protocol["rollout_initial_state_ids"])==len(protocol["rollout_seeds"])==20
            report["confirmation_protocol"]=protocol
            report["verified_checkpoint_sha256"]=checksum.hexdigest()
        saved=torch.load(args.checkpoint,map_location="cpu")
        cfg=dict(saved["config"])
        from adapters.baseline_causal_runtime import record_run
        record_run("act",out,cfg,training=False)
        if protocol:
            assert saved["step"]==protocol["training_step"]
            for key in ["rollout_horizon","settle_steps","execution_horizon","chunk_size","temporal_ensemble"]:
                assert cfg[key]==protocol[key], "Confirmation must preserve "+key
            cfg["rollout_initial_state_ids"]=protocol["rollout_initial_state_ids"]
            cfg["rollout_seeds"]=protocol["rollout_seeds"]
            cfg["purpose"]=protocol["purpose"]
        assert saved["step"] >= args.minimum_training_step, "Insufficient checkpoint updates"
        report["purpose"]=cfg.get("purpose",report["purpose"])
        report["checkpoint_selection"]=saved.get("selection","specified checkpoint")
        stats=saved["normalization"]
        report.update(config=cfg,checkpoint_training_step=saved["step"])
        model=build_act(chunk_size=cfg["chunk_size"],pretrained_backbone=False)
        model.load_state_dict(saved["model"],strict=True)
        del saved
        model=model.cuda().eval()
        report["timings"]["policy_ready_seconds"]=time.time()-start
        print("Policy loaded; creating environment",flush=True)
        from libero.libero import benchmark,get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        suite=benchmark.get_benchmark_dict()["libero_object"]()
        task=suite.get_task(0)
        env=OffScreenRenderEnv(
            bddl_file_name=str(Path(get_libero_path("bddl_files"))/task.problem_folder/task.bddl_file),
            camera_heights=128,camera_widths=128,render_gpu_device_id=0,control_freq=20)
        init_states=suite.get_task_init_states(0)
        report["available_initial_states"]=len(init_states)
        assert all(0<=i<len(init_states) for i in cfg["rollout_initial_state_ids"]), "Initial state unavailable; do not substitute"
        if protocol:
            assert len(init_states)>=protocol["required_initial_states"]
        report["task"]=task.name
        for initial_id,seed in zip(cfg["rollout_initial_state_ids"],cfg["rollout_seeds"]):
            if stop_requested:
                report["interrupted"]=True
                break
            np.random.seed(seed); torch.manual_seed(seed); env.seed(seed)
            env.reset()
            obs=env.set_init_state(init_states[initial_id])
            for _ in range(cfg["settle_steps"]):
                obs,_,_,_=env.step(np.zeros(7,dtype=np.float32))
            print("Starting rollout",initial_id,flush=True)
            frames=[obs["agentview_image"][::-1].copy()]
            trajectory=[]
            chunk=None
            success=False
            rollout_start=time.time()
            completed=False
            for t in range(cfg["rollout_horizon"]):
                if stop_requested:
                    report["interrupted"]=True
                    break
                if t%cfg["execution_horizon"]==0:
                    prepared=online_observation(obs,stats)
                    state=torch.from_numpy(prepared["state"]).unsqueeze(0).cuda()
                    images=torch.from_numpy(prepared["images"]).unsqueeze(0).cuda()
                    with torch.no_grad():
                        prediction=model(state,images,None)[0][0].cpu().numpy()
                    chunk=decode_action(prediction,stats)
                action=chunk[t%cfg["execution_horizon"]]
                obs,reward,done,_=env.step(action)
                assert np.isfinite(env.sim.data.qpos).all()
                trajectory.append({"step":t,"action":action.tolist(),
                                   "reward":float(reward),"done":bool(done)})
                if t%5==0:
                    frames.append(obs["agentview_image"][::-1].copy())
                success=bool(env.check_success())
                if success or done:
                    completed=True
                    break
            else:
                completed=True
            imageio.mimsave(str(out/("rollout_%d.gif"%initial_id)),frames,duration=0.25)
            (out/("rollout_%d.json"%initial_id)).write_text(json.dumps(trajectory))
            result={"initial_state_id":initial_id,"seed":seed,"steps":len(trajectory),
                    "success":success,"completed":completed,
                    "wall_seconds":time.time()-rollout_start}
            report["rollouts"].append(result)
            (out/"partial-result.json").write_text(json.dumps(report,indent=2))
            print("Rollout:",json.dumps(result),flush=True)
        report["execution_ok"]=(len(report["rollouts"])==len(cfg["rollout_initial_state_ids"])
                                and all(r["completed"] for r in report["rollouts"]))
        report["successes"]=sum(r["success"] for r in report["rollouts"] if r["completed"])
        report["completed_rollouts"]=sum(r["completed"] for r in report["rollouts"])
        n=report["completed_rollouts"]
        if n:
            import math
            p=report["successes"]/n
            z=1.959963984540054
            center=(p+z*z/(2*n))/(1+z*z/n)
            radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
            report["success_rate"]=p
            report["wilson95"]=[max(0,center-radius),min(1,center+radius)]
        report["note"]="Single-task, single-training-seed evaluation; incomplete trials are not counted as completed."
    except Exception as exc:
        report["error"]=repr(exc)
        import traceback
        traceback.print_exc()
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                report["close_error"]=repr(exc)
        report["wall_seconds"]=time.time()-start
        (out/"result.json").write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps(report,indent=2),flush=True)
    return 0 if report["execution_ok"] else 1

if __name__=="__main__":
    sys.exit(main())
