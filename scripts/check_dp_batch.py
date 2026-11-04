"""Single real-batch DP integration diagnostic, not a trained policy result."""
import json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
assert os.environ.get("SLURM_JOB_ID")
job=os.environ["SLURM_JOB_ID"]
out=ROOT/"artifacts/dp-batch"/job
out.mkdir(parents=True,exist_ok=True)
start=time.time()
report={"job_id":job,"passed":False,"purpose":"DP_single_batch_integration",
        "training_updates":0,"online_rollouts":0,"inference_steps":10}
try:
    import numpy as np
    import torch
    from adapters.dp_model import build_dp,make_normalizer,load_dp_state,assert_dp_placement
    from adapters.libero_sequence import LiberoSequenceDataset
    torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(4)
    assert torch.cuda.is_available()
    report.update(gpu=torch.cuda.get_device_name(0),torch=torch.__version__)
    path=Path.home()/"scratch/robot_manipulation/datasets/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5"
    split=json.loads((ROOT/"configs/act/split.json").read_text())
    normalizer=make_normalizer(path,split["train"])
    ds=LiberoSequenceDataset(path,split["train"],16,2)
    samples=[ds[0],ds[ds.ends[0]-1]]
    batch={"obs":{k:torch.from_numpy(np.stack([s["obs"][k] for s in samples])).cuda()
                  for k in ("agentview","wrist","agent_pos")},
           "action":torch.from_numpy(np.stack([s["action"] for s in samples])).cuda()}
    report["shapes"]={"obs":{k:list(v.shape) for k,v in batch["obs"].items()},
                      "action":list(batch["action"].shape)}
    # Attach CPU-created statistics before moving the complete module tree.
    policy=build_dp(pretrained=False,inference_steps=10)
    policy.set_normalizer(normalizer)
    policy=policy.cuda()
    assert_dp_placement(policy,batch["action"].device)
    report["initial_device_placement_passed"]=True
    reconstructed=policy.normalizer["action"].unnormalize(
        policy.normalizer["action"].normalize(batch["action"]))
    torch.testing.assert_close(reconstructed,batch["action"],rtol=1e-5,atol=1e-6)
    optimizer=torch.optim.AdamW(policy.parameters(),lr=1e-4)
    policy.train()
    optimizer.zero_grad(set_to_none=True)
    loss=policy.compute_loss(batch)
    assert torch.isfinite(loss).item()
    loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(policy.parameters(),1)
    assert torch.isfinite(norm).item() and norm.item()>0
    optimizer.step()
    report.update(loss=float(loss.detach()),gradient_norm=float(norm),training_updates=1)
    policy.eval()
    torch.manual_seed(123)
    with torch.no_grad():
        before=policy.predict_action(batch["obs"])["action"]
    assert tuple(before.shape)==(2,8,7) and torch.isfinite(before).all()
    folder=Path.home()/"scratch/robot_manipulation/dp-batch"/job
    folder.mkdir(parents=True,exist_ok=True)
    checkpoint=folder/"interface.pt"
    torch.save({"policy":{k:v.detach().cpu() for k,v in policy.state_dict().items()},
                "split":split,"source_commit":"5ba07ac6661db573af695b419a7947ecb704690f",
                "purpose":report["purpose"]},str(checkpoint))
    saved=torch.load(str(checkpoint),map_location="cpu")
    with torch.no_grad():
        next(policy.model.parameters()).add_(0.01)
    load_dp_state(policy,saved["policy"])
    assert_dp_placement(policy,batch["action"].device)
    report["reload_device_placement_passed"]=True
    torch.manual_seed(123)
    with torch.no_grad():
        after=policy.predict_action(batch["obs"])["action"]
    torch.testing.assert_close(before,after,rtol=1e-5,atol=1e-6)
    print("DP loss/backward/sampling/reload passed; checking live environment",flush=True)
    from libero.libero import benchmark,get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from PIL import Image
    suite=benchmark.get_benchmark_dict()["libero_object"]()
    task=suite.get_task(0)
    env=OffScreenRenderEnv(
        bddl_file_name=str(Path(get_libero_path("bddl_files"))/task.problem_folder/task.bddl_file),
        camera_heights=128,camera_widths=128,render_gpu_device_id=0)
    try:
        env.seed(0); env.reset()
        obs=env.set_init_state(suite.get_task_init_states(0)[0])
        live={"agentview":np.repeat(obs["agentview_image"].transpose(2,0,1)[None],2,axis=0).astype(np.float32)/255,
              "wrist":np.repeat(obs["robot0_eye_in_hand_image"].transpose(2,0,1)[None],2,axis=0).astype(np.float32)/255,
              "agent_pos":np.repeat(np.concatenate([obs["robot0_joint_pos"],obs["robot0_gripper_qpos"]])[None],2,axis=0).astype(np.float32)}
        live={k:torch.from_numpy(v).unsqueeze(0).cuda() for k,v in live.items()}
        with torch.no_grad():
            live_actions=policy.predict_action(live)["action"][0].cpu().numpy()
        assert live_actions.shape==(8,7) and np.isfinite(live_actions).all()
        for action in live_actions[:5]:
            obs,_,_,_=env.step(np.clip(action,-1,1))
            assert np.isfinite(env.sim.data.qpos).all()
        Image.fromarray(obs["agentview_image"][::-1]).save(out/"live_step5.png")
        report["live_environment_steps"]=5
    finally:
        env.close()
    report.update(passed=True,checkpoint=str(checkpoint),
                  checkpoint_bytes=checkpoint.stat().st_size,
                  reload_action_max_error=float((before-after).abs().max()),
                  max_gpu_memory_bytes=torch.cuda.max_memory_allocated())
except Exception as exc:
    report["error"]=repr(exc)
    import traceback
    traceback.print_exc()
finally:
    report["wall_seconds"]=time.time()-start
    (out/"result.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2),flush=True)
sys.exit(0 if report["passed"] else 1)
