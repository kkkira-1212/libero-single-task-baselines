"""Packaged frozen-DP inference/EGL smoke check; no task-success score or updates."""
import json,os,sys,random,hashlib,time
from collections import deque
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from PIL import Image
from adapters.dp_model import build_dp,load_dp_state,assert_dp_placement
from adapters.baseline_causal_runtime import record_run
from experiments.dp_run import live_obs
assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
job=os.environ['SLURM_JOB_ID'];out=ROOT/'artifacts/release-dp-smoke'/job;out.mkdir(parents=True,exist_ok=True)
report=dict(passed=False,job=job,training_updates=0,scope='Frozen-model packaged inference and10simulatoractions; not a benchmark score')
env=None;start=time.time()
try:
 c=json.loads((ROOT/'configs/baselines_causal/dp_seed0.json').read_text());record_run('dp',out,c,training=False)
 row=next(r for r in json.loads((ROOT/'results/runs.json').read_text())['runs'] if r['method']=='dp' and r['seed']==0 and r['setting']=='execute5')
 h=hashlib.sha256()
 with Path(row['checkpoint']).open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 assert h.hexdigest()==row['checkpoint_sha256']
 torch.set_num_threads(4);random.seed(20);np.random.seed(20);torch.manual_seed(20)
 saved=torch.load(row['checkpoint'],map_location='cpu');assert saved['config']==c and saved['step']==3000
 policy=build_dp(False,100);load_dp_state(policy,saved['ema_policy']);policy.cuda().eval();assert_dp_placement(policy,torch.device('cuda:0'));del saved
 from libero.libero import benchmark,get_libero_path
 from libero.libero.envs import OffScreenRenderEnv
 suite=benchmark.get_benchmark_dict()['libero_object']();task=suite.get_task(0)
 env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),camera_heights=128,camera_widths=128,render_gpu_device_id=0)
 env.seed(20);env.reset();obs=env.set_init_state(suite.get_task_init_states(0)[20])
 for _ in range(10):obs,_,_,_=env.step(np.zeros(7))
 history=deque([live_obs(obs)]*2,maxlen=2);frames=[Image.fromarray(obs['agentview_image'][::-1])];actions=[]
 for call in range(2):
  batch={k:torch.from_numpy(np.stack([h[k] for h in history])).unsqueeze(0).cuda() for k in history[0]}
  with torch.no_grad():pred=policy.predict_action(batch)['action'][0].cpu().numpy()
  assert pred.shape==(8,7) and np.isfinite(pred).all()
  for action in pred[:5]:
   action=np.clip(action,-1,1);obs,_,done,_=env.step(action)
   assert np.isfinite(env.sim.data.qpos).all();history.append(live_obs(obs));actions.append(action.tolist())
   if done:break
  frames.append(Image.fromarray(obs['agentview_image'][::-1]))
  if done:break
 frames[0].save(out/'smoke.gif',save_all=True,append_images=frames[1:],duration=250,loop=0)
 (out/'actions.json').write_text(json.dumps(actions))
 report.update(passed=True,simulator_actions=len(actions),policy_calls=call+1,gpu=torch.cuda.get_device_name(0),checkpoint_sha256=h.hexdigest(),frame_shape=list(obs['agentview_image'].shape),checkpoint_step=3000)
finally:
 if env is not None:env.close()
 report['wall_seconds']=time.time()-start
 (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report,indent=2),flush=True)
