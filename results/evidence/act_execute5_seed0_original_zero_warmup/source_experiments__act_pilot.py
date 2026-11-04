"""Bounded ACT pipeline pilot with training, resume verification and online rollouts."""
import argparse
import json
import os
from pathlib import Path
import random
import shutil
import signal
import socket
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
stop_requested=False
def request_stop(signum,frame):
    global stop_requested
    stop_requested=True

def main():
    global stop_requested
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",default="configs/act/pilot.json")
    parser.add_argument("--resume",default=None)
    args=parser.parse_args()
    assert os.environ.get("SLURM_JOB_ID"), "Use a compute allocation"
    signal.signal(signal.SIGUSR1,request_stop)
    signal.signal(signal.SIGTERM,request_stop)
    start=time.time()
    job=os.environ["SLURM_JOB_ID"]
    out=ROOT/"artifacts/act-pilot"/job
    out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((ROOT/args.config).read_text())
    report={"job_id":job,"host":socket.gethostname(),"purpose":cfg["purpose"],
            "execution_ok":False,"config":cfg,"resume_verified":False}
    env=None
    history=[]
    local=None
    try:
        from adapters.baseline_causal_runtime import record_run
        options=record_run("act",out,cfg,training=True)
        import numpy as np
        import torch
        from adapters.act_model import build_act,act_loss
        from adapters.libero_act import LiberoACTDataset,online_observation,decode_action
        seed=cfg["seed"]
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        torch.set_num_threads(4)
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False
        assert torch.cuda.is_available()
        report.update(gpu=torch.cuda.get_device_name(0),torch=torch.__version__,
                      cuda_build=torch.version.cuda)
        split=json.loads((ROOT/"configs/act/split.json").read_text())
        stats=json.loads((ROOT/"configs/act/normalization.json").read_text())
        interface=json.loads((ROOT/"configs/act/interface.json").read_text())
        interface["pretrained_backbone"]=cfg["pretrained_backbone"]
        interface["pretrained_note"]="Resolved from saved run config"
        assert not set(split["train"])&set(split["validation"])
        assert set(stats["training_episodes"])==set(split["train"])
        source=Path.home()/"scratch/robot_manipulation/datasets/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5"
        local=tempfile.TemporaryDirectory(prefix="act-pilot-"+job+"-",
                                         dir=os.environ.get("SLURM_TMPDIR") or None)
        data_path=Path(local.name)/"demo.hdf5"
        shutil.copyfile(str(source),str(data_path))
        assert data_path.stat().st_size==source.stat().st_size
        if cfg.get("dataset_sha256"):
            import hashlib
            checksum=hashlib.sha256()
            with data_path.open("rb") as stream:
                for block in iter(lambda:stream.read(8*1024*1024),b""):checksum.update(block)
            assert checksum.hexdigest()==cfg["dataset_sha256"]
        train=LiberoACTDataset(data_path,split["train"],stats,cfg["chunk_size"],**options)
        validation=LiberoACTDataset(data_path,split["validation"],stats,cfg["chunk_size"],**options)
        report.update(train_steps=len(train),validation_steps=len(validation),
                      act_source_commit="742c753c0d4a5d87076c8f69e5628c79a8cc5488")
        model=build_act(chunk_size=cfg["chunk_size"],
                        pretrained_backbone=cfg["pretrained_backbone"]).cuda()
        optimizer=torch.optim.AdamW([
            {"params":[p for n,p in model.named_parameters() if "backbone" not in n],"lr":cfg["lr"]},
            {"params":[p for n,p in model.named_parameters() if "backbone" in n],"lr":cfg["backbone_lr"]}
        ],weight_decay=1e-4)
        def rng_state():
            return {"python":random.getstate(),"numpy":np.random.get_state(),
                    "torch":torch.get_rng_state(),"cuda":torch.cuda.get_rng_state_all()}
        def restore_rng(state):
            random.setstate(state["python"]); np.random.set_state(state["numpy"])
            torch.set_rng_state(state["torch"]); torch.cuda.set_rng_state_all(state["cuda"])
        def batch_samples(dataset,indices):
            samples=[dataset[int(i)] for i in indices]
            return {k:torch.from_numpy(np.stack([s[k] for s in samples])).cuda()
                    for k in ("images","state","actions","is_pad")}
        val_indices=[]
        for ep,n in enumerate(validation.sample_lengths):
            offset=validation.ends[ep-1] if ep else 0
            val_indices.extend(offset+int(t) for t in np.linspace(0,n-1,cfg["validation_samples_per_demo"]))
        def validation_l1():
            model.eval()
            total=0.0; count=0
            with torch.no_grad():
                for offset in range(0,len(val_indices),cfg["batch_size"]):
                    b=batch_samples(validation,val_indices[offset:offset+cfg["batch_size"]])
                    prediction=model(b["state"],b["images"],None)[0]
                    mask=(~b["is_pad"]).unsqueeze(-1).expand_as(prediction)
                    total+=float(((prediction-b["actions"]).abs()*mask).sum())
                    count+=int(mask.sum())
            model.train()
            return total/count
        checkpoint_dir=Path.home()/"scratch/robot_manipulation/act-pilot"/job
        checkpoint_dir.mkdir(parents=True,exist_ok=True)
        checkpoint=checkpoint_dir/"latest.pt"
        best_policy=checkpoint_dir/"best-policy.pt"
        best_validation=float("inf")
        def save_best(step,val):
            nonlocal best_validation
            if val < best_validation:
                payload={"model":model.state_dict(),"step":step,"config":cfg,
                         "split":split,"normalization":stats,"interface":interface,
                         "source_commit":report["act_source_commit"],
                         "validation_l1":val,"selection":"minimum_sampled_validation_L1"}
                temporary=best_policy.with_suffix(".pt.partial")
                torch.save(payload,str(temporary))
                os.replace(str(temporary),str(best_policy))
                best_validation=val
        def save_checkpoint(step):
            payload={"model":model.state_dict(),"optimizer":optimizer.state_dict(),
                     "rng":rng_state(),"step":step,"config":cfg,"split":split,
                     "normalization":stats,"interface":interface,
                     "source_commit":report["act_source_commit"],"history":history}
            temporary=checkpoint.with_suffix(".pt.partial")
            torch.save(payload,str(temporary))
            os.replace(str(temporary),str(checkpoint))
        def load_checkpoint(path):
            saved=torch.load(str(path),map_location="cpu")
            assert saved["config"]==cfg and saved["split"]==split
            assert saved["normalization"]==stats and saved["interface"]==interface
            model.load_state_dict(saved["model"],strict=True)
            optimizer.load_state_dict(saved["optimizer"])
            restore_rng(saved["rng"])
            return saved["step"],saved["history"]
        step=0
        if args.resume:
            step,history=load_checkpoint(args.resume)
            report["checkpoint_restored"] = str(args.resume)
        start_step = step
        def update():
            model.train()
            indices=np.random.randint(0,len(train),size=cfg["batch_size"])
            b=batch_samples(train,indices)
            optimizer.zero_grad(set_to_none=True)
            prediction,_,(mu,logvar)=model(b["state"],b["images"],None,b["actions"],b["is_pad"])
            losses=act_loss(prediction,b["actions"],b["is_pad"],mu,logvar,cfg["kl_weight"])
            assert torch.isfinite(losses["loss"]).item()
            losses["loss"].backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg["grad_clip"])
            assert torch.isfinite(norm).item()
            optimizer.step()
            return {"indices":indices.tolist(),**{k:float(v.detach()) for k,v in losses.items()},
                    "grad_norm":float(norm)}
        initial=validation_l1()
        history.append({"step":step,"validation_prior_valid_l1":initial})
        save_best(step,initial)
        print("Initial validation:",initial,flush=True)
        while step<cfg["updates"] and not stop_requested:
            if time.time()-start>cfg["wall_budget_before_final_checkpoint_seconds"]:
                report["stopped_for_wall_budget"]=True
                break
            metrics=update(); step+=1
            if step%cfg["checkpoint_interval"]==0:
                val=validation_l1()
                row={"step":step,"validation_prior_valid_l1":val,
                     **{k:v for k,v in metrics.items() if k!="indices"}}
                history.append(row)
                print(json.dumps(row),flush=True)
                save_checkpoint(step)
                save_best(step,val)
                if cfg.get("verify_resume",True) and not report["resume_verified"]:
                    # Execute one reference update, restore every state, replay same update.
                    reference_metrics=update()
                    reference={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                    restored_step,_=load_checkpoint(checkpoint)
                    replay_metrics=update()
                    assert restored_step==step
                    assert reference_metrics["indices"]==replay_metrics["indices"]
                    max_error=0.0
                    for name,tensor in model.state_dict().items():
                        actual=tensor.detach().cpu()
                        torch.testing.assert_close(actual,reference[name],rtol=1e-5,atol=1e-6)
                        max_error=max(max_error,float((actual-reference[name]).abs().max()))
                    report["resume_verified"]=True
                    report["resume_next_update_max_parameter_error"]=max_error
                    step+=1
                    del reference
                    print("Optimizer and RNG replay verified",flush=True)
        final=validation_l1()
        report.update(updates_completed=step,updates_this_run=step-start_step,initial_validation_l1=initial,
                      final_validation_l1=final,completed_target=step>=cfg["updates"],
                      validation_metric_scope="%d fixed timesteps across 10 held-out demos"%len(val_indices))
        save_checkpoint(step)
        save_best(step,final)
        report.update(checkpoint=str(checkpoint),checkpoint_bytes=checkpoint.stat().st_size,
                      best_policy=str(best_policy),best_validation_l1=best_validation)
        if cfg.get("training_only",False):
            report["execution_ok"]=step>=cfg["updates"] and not stop_requested
            report["note"]="Training stage only; online evaluation is a separate dependent job"
            return 0 if report["execution_ok"] else 1
        # Reload final weights before online inference; a clean optimizer state is also present.
        load_checkpoint(checkpoint)
        print("Starting online evaluation",flush=True)
        from libero.libero import benchmark,get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        import h5py
        import imageio
        suite=benchmark.get_benchmark_dict()["libero_object"]()
        task=suite.get_task(0)
        bddl=Path(get_libero_path("bddl_files"))/task.problem_folder/task.bddl_file
        env=OffScreenRenderEnv(bddl_file_name=str(bddl),
                               camera_heights=128,camera_widths=128,
                               render_gpu_device_id=0,control_freq=20)
        env.seed(seed); env.reset()
        # Replay a stored simulator state and compare joint/gripper fields directly.
        with h5py.File(data_path,"r") as f:
            d=f["data"][split["train"][0]]
            # Official create_dataset.py stores obs[t] after action[t], states[t] before it.
            # Existing same-index behavior-cloning convention is deliberately unchanged.
            parity_obs=env.set_init_state(d["states"][1])
            online=online_observation(parity_obs,stats)
            expected=train[0]
            state_error=float(np.max(np.abs(online["state"]-expected["state"])))
            raw_mae={key:float(np.mean(np.abs(
                parity_obs[online_key].astype(np.float32)-d["obs"][key][0].astype(np.float32))))
                for key,online_key in [("agentview_rgb","agentview_image"),
                                      ("eye_in_hand_rgb","robot0_eye_in_hand_image")]}
            flipped_mae={key:float(np.mean(np.abs(
                parity_obs[online_key][::-1].astype(np.float32)-d["obs"][key][0].astype(np.float32))))
                for key,online_key in [("agentview_rgb","agentview_image"),
                                      ("eye_in_hand_rgb","robot0_eye_in_hand_image")]}
            report["stored_state_parity"]={"normalized_proprio_max_error":state_error,
                                          "camera_raw_mae":raw_mae,
                                          "camera_flipped_mae":flipped_mae,
                                          "stored_observation_index":0,
                                          "replayed_state_index":1}
            assert all(raw_mae[k] <= flipped_mae[k] for k in raw_mae), "Check camera orientation before rollouts"
            assert state_error<1e-4,"Stored vs online proprioception mismatch"
        model.eval()
        rollouts=[]
        init_states=suite.get_task_init_states(0)
        for initial_id,rollout_seed in zip(cfg["rollout_initial_state_ids"],cfg["rollout_seeds"]):
            if stop_requested or time.time()-start>1500:
                report["rollouts_stopped_for_time"]=True
                break
            env.seed(rollout_seed); env.reset()
            obs=env.set_init_state(init_states[initial_id])
            for _ in range(cfg["settle_steps"]):
                obs,_,_,_=env.step(np.zeros(7,dtype=np.float32))
            frames=[]; trajectory=[]; chunk=None; success=False
            for t in range(cfg["rollout_horizon"]):
                if t%cfg["execution_horizon"]==0:
                    prepared=online_observation(obs,stats)
                    state=torch.from_numpy(prepared["state"]).unsqueeze(0).cuda()
                    images=torch.from_numpy(prepared["images"]).unsqueeze(0).cuda()
                    with torch.no_grad():
                        predicted=model(state,images,None)[0][0].cpu().numpy()
                    chunk=decode_action(predicted,stats)
                action=chunk[t%cfg["execution_horizon"]]
                obs,reward,done,_=env.step(action)
                assert np.isfinite(env.sim.data.qpos).all()
                trajectory.append({"step":t,"action":action.tolist(),"reward":float(reward),"done":bool(done)})
                if t%5==0:
                    frames.append(obs["agentview_image"][::-1].copy())
                success=bool(env.check_success())
                if success or done:
                    break
            imageio.mimsave(str(out/("rollout_%d.gif"%initial_id)),frames,duration=0.25)
            (out/("rollout_%d.json"%initial_id)).write_text(json.dumps(trajectory))
            rollouts.append({"initial_state_id":initial_id,"seed":rollout_seed,
                             "steps":len(trajectory),"success":success})
            print("Pilot rollout:",rollouts[-1],flush=True)
        report.update(rollouts=rollouts,
                      execution_ok=len(rollouts)==len(cfg["rollout_initial_state_ids"]),
                      note="Pilot diagnostics only; no credible success-rate claim from two short rollouts",
                      max_gpu_memory_bytes=torch.cuda.max_memory_allocated())
    except Exception as exc:
        report["error"]=repr(exc)
        import traceback
        traceback.print_exc()
    finally:
        if env is not None:
            env.close()
        if local is not None:
            local.cleanup()
        report["wall_seconds"]=time.time()-start
        (out/"history.json").write_text(json.dumps(history,indent=2)+"\n")
        (out/"result.json").write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps(report,indent=2),flush=True)
    return 0 if report["execution_ok"] else 1

if __name__=="__main__":
    sys.exit(main())
