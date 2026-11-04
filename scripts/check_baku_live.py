"""Headless BAKU/LIBERO integration; no training or success-rate estimate."""
import json,os,random,sys,time
from pathlib import Path
import h5py,numpy as np,torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"third_party/BAKU/baku"))
from adapters.baku_live import online_observation,identity_stats
assert os.environ.get("SLURM_JOB_ID")
job=os.environ["SLURM_JOB_ID"];out=ROOT/"artifacts/baku-live-check"/job
out.mkdir(parents=True,exist_ok=True)
report={"passed":False,"training_updates_this_run":0,"live_environment_steps":0,"online_rollouts":0}
start=time.time();env=None
try:
    from agent.baku import BCAgent
    from libero.libero import benchmark,get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    torch.set_num_threads(4);random.seed(0);np.random.seed(0);torch.manual_seed(0)
    assert torch.cuda.is_available()
    checkpoint=Path.home()/"scratch/robot_manipulation/baku-model-check/14354835/interface.pt"
    saved=torch.load(checkpoint,map_location="cpu")
    assert saved["source_commit"]=="47e367059191de30ac4b6353ff5615fdef11d8c5"
    agent=BCAgent(**saved["model_config"]);agent.load_snapshot(saved["snapshot"],eval=True)
    report.update(checkpoint=str(checkpoint),checkpoint_training_updates=saved["training_updates"],gpu=torch.cuda.get_device_name(0))
    del saved
    prep=json.loads((ROOT/"artifacts/baku-prepare/14354211/result.json").read_text())
    embedding=np.load(Path(prep["output_root"])/"task_embedding.npy")
    prompt={"task_emb":embedding}
    suite=benchmark.get_benchmark_dict()["libero_object"]();task=suite.get_task(0)
    assert task.name=="pick_up_the_alphabet_soup_and_place_it_in_the_basket"
    env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path("bddl_files"))/task.problem_folder/task.bddl_file),
                          camera_heights=128,camera_widths=128,render_gpu_device_id=0)
    env.seed(0);env.reset()
    path=Path.home()/"scratch/robot_manipulation/datasets/libero_object"/(task.name+"_demo.hdf5")
    with h5py.File(path,"r") as f:
        # Dataset stores post-action obs: obs[0] matches states[1], as verified for ACT.
        d=f["data/demo_28"];obs=env.set_init_state(d["states"][1])
        prepared=online_observation(obs)
        expected=np.concatenate([d["obs/joint_states"][0],d["obs/gripper_states"][0]]).astype(np.float32)
        error=float(np.max(np.abs(prepared["proprioceptive"]-expected)))
        np.testing.assert_allclose(prepared["proprioceptive"],expected,rtol=0,atol=1e-6)
        report["stored_state_proprio_max_error"]=error
        image_errors={}
        for key,source in (("pixels","agentview_rgb"),("pixels_egocentric","eye_in_hand_rgb")):
            expected_image=d["obs"][source][0][::-1].transpose(2,0,1)
            mae=float(np.mean(np.abs(prepared[key].astype(float)-expected_image.astype(float))))
            # Rendering differences across machines are recorded, not required pixel-identical.
            assert mae<15,("Unexpected image orientation/render mismatch",key,mae)
            image_errors[key]=mae
        report["flipped_image_mae"]=image_errors
    obs=env.set_init_state(suite.get_task_init_states(0)[20])
    for _ in range(10):obs,_,_,_=env.step(np.zeros(7))
    agent.buffer_reset()
    frames=[Image.fromarray(obs["agentview_image"][::-1])];trajectory=[]
    with torch.no_grad():
        for step in range(30):
            prepared=online_observation(obs)
            action=np.asarray(agent.act(prepared,prompt,identity_stats(),step,step,eval_mode=True))
            assert action.shape==(7,) and np.isfinite(action).all()
            executed=np.clip(action,-1,1)
            obs,reward,done,_=env.step(executed)
            assert np.isfinite(env.sim.data.qpos).all()
            trajectory.append({"step":step,"action":action.tolist(),"executed_action":executed.tolist(),
                               "reward":float(reward),"done":bool(done),"success":bool(env.check_success())})
            frames.append(Image.fromarray(obs["agentview_image"][::-1]))
            report["live_environment_steps"]=step+1
            if done:break
    assert report["live_environment_steps"]==30,"Unexpected early termination"
    frames[0].save(out/"live.gif",save_all=True,append_images=frames[1:],duration=50,loop=0)
    Image.fromarray(obs["robot0_eye_in_hand_image"][::-1]).save(out/"wrist_final.png")
    (out/"actions.json").write_text(json.dumps(trajectory,indent=2)+"\n")
    report.update(passed=True,note="30stepinterfacecheck with1-update model;not a complete rollout or success-rate evaluation.")
except Exception as exc:
    report["error"]=repr(exc)
    import traceback;traceback.print_exc()
finally:
    if env is not None:env.close()
    report["wall_seconds"]=time.time()-start
    (out/"result.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2),flush=True)
sys.exit(0 if report["passed"] else 1)
