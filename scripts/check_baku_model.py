"""Combined real-batch author BAKU update, prediction and checkpoint test."""
import json,os,random,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"third_party/BAKU/baku"))
assert os.environ.get("SLURM_JOB_ID")
job=os.environ["SLURM_JOB_ID"];out=ROOT/"artifacts/baku-model-check"/job
out.mkdir(parents=True,exist_ok=True)
report={"passed":False,"training_updates":0,"online_rollouts":0,"live_environment_steps":0,
        "purpose":"BAKU_author_model_real_batch_integration"}
start=time.time()
try:
    from agent.baku import BCAgent
    from read_data.libero import BCDataset
    random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.set_num_threads(4)
    assert torch.cuda.is_available()
    prep=json.loads((ROOT/"artifacts/baku-prepare/14354211/result.json").read_text())
    task="pick_up_the_alphabet_soup_and_place_it_in_the_basket"
    ds=BCDataset(path=str(Path(prep["output_root"])/"train"),suite="libero_object",
                 scenes=["object"],tasks=[{"object":[task]}],num_demos_per_task=40,
                 obs_type="pixels",history=False,history_len=10,prompt="text",
                 temporal_agg=True,num_queries=10,img_size=128)
    batch=next(iter(torch.utils.data.DataLoader(ds,batch_size=2,num_workers=0)))
    report["shapes"]={k:list(v.shape) for k,v in batch.items()}
    assert batch["actions"].shape==(2,1,10,7)
    kwargs=dict(obs_shape={"pixels":(3,128,128),"pixels_egocentric":(3,128,128),"proprioceptive":(9,)},
                action_shape=(7,),device="cuda",lr=1e-4,hidden_dim=256,
                stddev_schedule=0.1,stddev_clip=0.3,use_tb=True,augment=True,
                obs_type="pixels",encoder_type="resnet",policy_type="gpt",policy_head="deterministic",
                pixel_keys=["pixels","pixels_egocentric"],proprio_key="proprioceptive",feature_key="features",
                use_proprio=True,train_encoder=True,norm=False,history=False,history_len=10,
                eval_history_len=5,separate_encoders=False,temporal_agg=True,
                max_episode_len=600,num_queries=10,prompt="text",use_language=True,film=True)
    agent=BCAgent(**kwargs);report["model_config"]=kwargs
    before=next(agent.actor.parameters()).detach().clone()
    copied={k:v.clone() for k,v in batch.items()}
    metrics=agent.update(iter([copied]),step=0)
    assert metrics and all(np.isfinite(v) for v in metrics.values()),metrics
    for module in (agent.actor,agent.encoder,agent.proprio_projector,agent.language_projector):
        assert all(torch.isfinite(p).all() for p in module.parameters())
        grads=[p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
    delta=float((before-next(agent.actor.parameters())).abs().max())
    assert delta>0
    report.update(training_updates=1,metrics=metrics,actor_parameter_max_change=delta)
    obs={k:np.rint(batch[k][0,-1].numpy()*255).astype(np.uint8) for k in ("pixels","pixels_egocentric")}
    obs["proprioceptive"]=batch["proprioceptive"][0,-1].numpy()*(1+1e-5)
    prompt={"task_emb":batch["task_emb"][0].numpy()}
    def predict():
        agent.train(False);agent.buffer_reset()
        with torch.no_grad():
            return agent.act({k:v.copy() for k,v in obs.items()},prompt,ds.stats,0,1,eval_mode=True)
    action=predict()
    assert action.shape==(7,) and np.isfinite(action).all()
    folder=Path.home()/"scratch/robot_manipulation/baku-model-check"/job
    folder.mkdir(parents=True,exist_ok=True);checkpoint=folder/"interface.pt"
    torch.save({"snapshot":agent.save_snapshot(),"model_config":kwargs,
                "source_commit":"47e367059191de30ac4b6353ff5615fdef11d8c5",
                "training_updates":1,"purpose":report["purpose"]},checkpoint)
    with torch.no_grad():next(agent.actor.parameters()).add_(0.01)
    saved=torch.load(checkpoint,map_location="cpu")
    agent.load_snapshot(saved["snapshot"],eval=True)
    restored=predict()
    np.testing.assert_allclose(action,restored,rtol=1e-5,atol=1e-6)
    report.update(passed=True,checkpoint=str(checkpoint),reload_action_max_error=float(np.max(np.abs(action-restored))),
                  gpu=torch.cuda.get_device_name(0),note="Offline image-based act only; live LIBERO integration and trained success still pending.")
except Exception as exc:
    report["error"]=repr(exc)
    import traceback;traceback.print_exc()
finally:
    report["wall_seconds"]=time.time()-start
    (out/"result.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2),flush=True)
sys.exit(0 if report["passed"] else 1)
