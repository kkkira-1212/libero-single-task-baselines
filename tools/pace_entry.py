"""Explicit entry points for the archived causal-v1 experiments on PACE."""
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from prepare_runtime import prepare
p=argparse.ArgumentParser();p.add_argument('mode',choices=['data-check','gate','train','eval'])
p.add_argument('method',choices=['act','dp','baku','a2a']);p.add_argument('seed',type=int,choices=[0,1])
p.add_argument('--checkpoint');p.add_argument('--config');p.add_argument('--protocol');p.add_argument('--gate-job');p.add_argument('--execute-one',action='store_true')
a=p.parse_args()
if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Submit with sbatch or use a compute allocation')
if a.execute_one and a.method!='dp':p.error('--execute-one is a DP evaluation override')
if a.execute_one and a.mode!='eval':p.error('--execute-one requires eval')
prepare();os.chdir(ROOT)
method=a.method
if method=='a2a':
 cfg=a.config or ('configs/a2a/train_timing_causal_seed0_10000.json' if a.seed==0 else 'configs/a2a/train_causal_seed1_10000.json')
 if a.mode in ('data-check','gate'):
  cmd=['scripts/check_a2a_causal_integration.py']+(['--data-only'] if a.mode=='data-check' else [])
 elif a.mode=='train':
  cmd=['experiments/a2a_run.py','train','--config',cfg,'--alignment-gate-job',a.gate_job or '14468575']
 else:
  if a.checkpoint:p.error('A2A frozen evaluation reads the checkpoint from the frozen manifest; make an explicit manifest copy to relocate')
  if a.config:p.error('A2A frozen evaluation uses the manifest original training config')
  if a.protocol:protocol=a.protocol
  else:
   target=ROOT/'artifacts'/('release-a2a-seed%s-original.json'%a.seed)
   frozen=json.loads((ROOT/('configs/a2a/frozen_causal_seed%s_10000.json'%a.seed)).read_text())
   target.write_text(json.dumps(dict(purpose='release_original_zero_warmup',sampling_steps=6,execution_horizon=1,checkpoint_sha256=frozen['checkpoint_sha256'])))
   protocol=str(target)
  cmd=['experiments/a2a_confirm_eval.py','--frozen-manifest','configs/a2a/frozen_causal_seed%s_10000.json'%a.seed,'--protocol',protocol]
else:
 group=json.loads((ROOT/'configs/baselines_causal/manifest_2026-10-27.json').read_text())['methods'][method]
 cfg=a.config or group['runs'][str(a.seed)]['config']
 os.environ['CAUSAL_GATE_JOB']=a.gate_job or group['gate_job']
 if a.mode in ('data-check','gate'):
  cmd=['scripts/check_baseline_causal.py','--method',method]+(['--data-only'] if a.mode=='data-check' else [])
 elif a.mode=='train':
  cmd=['experiments/act_pilot.py','--config',cfg] if method=='act' else ['experiments/'+method+'_run.py','train','--config',cfg]
 else:
  runs=json.loads((ROOT/'results/runs.json').read_text())['runs']
  row=next(r for r in runs if r['method']==method and r['seed']==a.seed and r['condition']=='original_zero_warmup' and not r.get('secondary'))
  checkpoint=a.checkpoint or row['checkpoint']
  if method=='act':
   if a.protocol:p.error('ACT custom protocol relocation is a separate advanced workflow; use archived original conditions')
   cmd=['experiments/act_eval.py','--checkpoint',checkpoint,'--minimum-training-step','10000']
  else:
   cmd=['experiments/'+method+'_run.py','eval','--config',cfg,'--checkpoint',checkpoint]
   protocol=a.protocol
   if a.execute_one:
    if protocol:p.error('Choose --execute-one or --protocol')
    target=ROOT/'artifacts'/('release-dp-seed%s-execute1.json'%a.seed)
    target.write_text(json.dumps(dict(purpose='release_frozen_execute1',execution_horizon=1,checkpoint_sha256=row['checkpoint_sha256'])))
    protocol=str(target)
   if protocol:cmd+=['--eval-protocol',protocol]
print('Running',sys.executable,*cmd,flush=True)
subprocess.run([sys.executable,'-u']+cmd,check=True,cwd=ROOT)
