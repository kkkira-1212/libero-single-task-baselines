"""Bundled author A2A loss/update/reload and LIBERO integration."""
import hashlib,json,os,random,sys,time
from collections import deque
from pathlib import Path
import numpy as np,torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
assert os.environ.get("SLURM_JOB_ID")
job=os.environ["SLURM_JOB_ID"];out=ROOT/"artifacts/a2a-integration"/job;out.mkdir(parents=True,exist_ok=True)
report={"passed":False,"training_updates":0,"online_rollouts":0,"sampling_steps":6,
        "purpose":"A2A_single_task_adapter_integration"};start=time.time();env=None
try:
 from adapters.a2a_model import build_a2a,make_normalizer,load_state,assert_placement
 from adapters.libero_sequence import LiberoSequenceDataset
 random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.set_num_threads(4)
 assert torch.cuda.is_available()
 path=Path.home()/"scratch/robot_manipulation/datasets/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5"
 h=hashlib.sha256()
 with path.open("rb") as f:
  for block in iter(lambda:f.read(8*1024*1024),b""):h.update(block)
 split=json.loads((ROOT/"configs/act/split.json").read_text());assert h.hexdigest()==split["dataset_sha256"]
 ds=LiberoSequenceDataset(path,split["train"],16,8)
 samples=[ds[0],ds[64]]
 batch={"obs":{k:torch.from_numpy(np.stack([s["obs"][k] for s in samples])).cuda() for k in ("agentview","wrist","agent_pos")},
        "action":torch.from_numpy(np.stack([s["action"] for s in samples])).cuda()}
 report["shapes"]={"obs":{k:list(v.shape) for k,v in batch["obs"].items()},"action":list(batch["action"].shape)}
 policy=build_a2a(6);policy.set_normalizer(make_normalizer(path,split["train"]));policy.cuda()
 assert_placement(policy,torch.device("cuda:0"))
 assert policy.history_action_encoder.action_dim==9 and policy.action_encoder.action_dim==7
 optimizer=torch.optim.AdamW(policy.parameters(),lr=1e-4)
 policy.train();optimizer.zero_grad(set_to_none=True)
 loss=policy.compute_loss(batch);assert torch.isfinite(loss)
 loss.backward()
 norm=torch.nn.utils.clip_grad_norm_(policy.parameters(),1.0);assert torch.isfinite(norm) and norm>0
 for name,module in (("history",policy.history_action_encoder),("future",policy.action_encoder),("decoder",policy.action_decoder),("flow",policy.flow_net)):
  grads=[p.grad for p in module.parameters() if p.grad is not None]
  assert grads and all(torch.isfinite(g).all() for g in grads),name
 optimizer.step();report.update(training_updates=1,loss=float(loss.detach()),gradient_norm=float(norm))
 policy.eval()
 with torch.no_grad():before=policy.predict_action(batch["obs"])["action"]
 assert before.shape==(2,8,7) and torch.isfinite(before).all()
 folder=Path.home()/"scratch/robot_manipulation/a2a-integration"/job;folder.mkdir(parents=True,exist_ok=True)
 checkpoint=folder/"interface.pt"
 torch.save({"policy":{k:v.detach().cpu() for k,v in policy.state_dict().items()},
            "source_commit":"a5792ecf4e7f8fa4d85fe66ea9a50618138f925c","split":split,
            "purpose":report["purpose"],"training_updates":1,"sampling_steps":6},checkpoint)
 with torch.no_grad():next(policy.flow_net.parameters()).add_(0.01)
 load_state(policy,torch.load(checkpoint,map_location="cpu")["policy"])
 assert_placement(policy,torch.device("cuda:0"))
 with torch.no_grad():after=policy.predict_action(batch["obs"])["action"]
 torch.testing.assert_close(before,after,rtol=1e-5,atol=1e-6)
 report["reload_action_max_error"]=float((before-after).abs().max())
 policy.num_sampling_steps=1
 with torch.no_grad():one=policy.predict_action(batch["obs"])["action"]
 assert one.shape==(2,8,7) and torch.isfinite(one).all()
 policy.num_sampling_steps=6;report["single_step_output_shape_finite_passed"]=True
 from libero.libero import benchmark,get_libero_path
 from libero.libero.envs import OffScreenRenderEnv
 suite=benchmark.get_benchmark_dict()["libero_object"]();task=suite.get_task(0)
 env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path("bddl_files"))/task.problem_folder/task.bddl_file),
                       camera_heights=128,camera_widths=128,render_gpu_device_id=0)
 def obs_array(obs):
  return {"agentview":obs["agentview_image"].transpose(2,0,1).astype(np.float32)/255,
          "wrist":obs["robot0_eye_in_hand_image"].transpose(2,0,1).astype(np.float32)/255,
          "agent_pos":np.concatenate([obs["robot0_joint_pos"],obs["robot0_gripper_qpos"]]).astype(np.float32)}
 env.seed(0);env.reset();obs=env.set_init_state(suite.get_task_init_states(0)[20])
 for _ in range(10):obs,_,_,_=env.step(np.zeros(7))
 history=deque([obs_array(obs)]*8,maxlen=8);frames=[Image.fromarray(obs["agentview_image"][::-1])];actions=[]
 with torch.no_grad():
  for step in range(30):
   live={k:torch.from_numpy(np.stack([a[k] for a in history])).unsqueeze(0).cuda() for k in history[0]}
   action=policy.predict_action(live)["action"][0,0].cpu().numpy();assert np.isfinite(action).all()
   executed=np.clip(action,-1,1);obs,reward,done,_=env.step(executed)
   assert np.isfinite(env.sim.data.qpos).all()
   history.append(obs_array(obs));frames.append(Image.fromarray(obs["agentview_image"][::-1]))
   actions.append(executed.tolist())
   if done:break
 assert len(actions)==30
 frames[0].save(out/"live.gif",save_all=True,append_images=frames[1:],duration=50,loop=0)
 (out/"actions.json").write_text(json.dumps(actions)+"\n")
 report.update(passed=True,live_environment_steps=30,checkpoint=str(checkpoint),gpu=torch.cuda.get_device_name(0),
               note="One update,6stepflow integration and1step finite-output check only;no learned success/latency claims.")
except Exception as exc:
 report["error"]=repr(exc)
 import traceback;traceback.print_exc()
finally:
 if env is not None:env.close()
 report["wall_seconds"]=time.time()-start
 (out/"result.json").write_text(json.dumps(report,indent=2)+"\n")
 print(json.dumps(report,indent=2),flush=True)
sys.exit(0 if report["passed"] else 1)
