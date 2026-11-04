"""Independent standard-library audit of the exported stage-one evidence."""
import csv,hashlib,json,math
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def read(p):return json.loads((R/p).read_text())
def verify():
 runs=read('results/runs.json')['runs'];checks=[];total=0
 split=read('configs/act/split.json')
 assert len(split['train'])==40 and len(split['validation'])==10
 assert not set(split['train'])&set(split['validation'])
 csv_rows=list(csv.DictReader((R/'results/summary.csv').open()))
 assert len(csv_rows)==len(runs)
 for row,tab in zip(runs,csv_rows):
  report=read(row['evidence']);folder=(R/row['evidence']).parent
  assert report.get('completed',report.get('execution_ok',False))
  records=report['rollouts'];n=len(records);successes=sum(r['success'] for r in records)
  assert n==row['rollouts'] and successes==row['successes']==report['successes']
  assert successes/n==row['success_rate']==report['success_rate']
  assert int(tab['successes'])==successes and int(tab['rollouts'])==n and float(tab['success_rate'])==successes/n
  assert int(tab['seed'])==row['seed'] and tab['method']==row['method'] and tab['condition']==row['condition']
  assert [r['initial_state_id'] for r in records]==row['initial_state_ids']
  assert len(set(row['initial_state_ids']))==n
  for r in records:
   assert r['completed'] and r['seed']==r['initial_state_id']
   file='rollout_%s.json'%r['initial_state_id'] if row['method']=='act' else 'trial_%s_actions.json'%r['initial_state_id']
   actions=read(folder.relative_to(R)/file)
   assert len(actions)==r['steps'] and 0<len(actions)<=600
   for action in actions:
    a=action['action'] if isinstance(action,dict) else action
    assert len(a)==7 and all(math.isfinite(x) and -1<=x<=1 for x in a)
  config=report['config']
  assert report['checkpoint_training_step']==row['updates']==config['updates']
  assert config['dataset_sha256']==split['dataset_sha256']
  assert config['sample_contract']==('libero-same-index-v0' if row['condition']=='timing_ablation_control' else 'libero-poststep-next-command-v1')
  assert config['settle_steps']==10
  if row['condition']=='diagnostic_open_warmup':assert report['evaluation_protocol']['settle_gripper_command']==-1
  elif 'evaluation_protocol' in report:assert report['evaluation_protocol'].get('settle_gripper_command',0)==0
  if row['method']=='dp':assert config['inference_steps']==100
  if row['method']=='a2a':assert report.get('sampling_steps',config['inference_steps'])==(1 if row['setting']=='flow1_execute1' else 6)
  protocol=folder/'causal_protocol.json'
  if protocol.exists():
   pr=json.loads(protocol.read_text())
   for source,sha in pr['source_sha256'].items():
    archived=folder/('source_'+source.replace('/','__'))
    assert hashlib.sha256(archived.read_bytes()).hexdigest()==sha,archived
  if report.get('checkpoint_sha256'):assert report['checkpoint_sha256']==row['checkpoint_sha256']
  assert len(row['checkpoint_sha256'])==64
  for v in row['representative_videos']:
   assert (R/v['path']).read_bytes()[:6] in (b'GIF87a',b'GIF89a')
  checks.append(dict(method=row['method'],seed=row['seed'],condition=row['condition'],setting=row['setting'],successes=successes,rollouts=n,passed=True));total+=n
 manifest=R/'results/file_manifest.json'
 if manifest.exists():
  for path,sha in json.loads(manifest.read_text()).items():assert hashlib.sha256((R/path).read_bytes()).hexdigest()==sha,path
 result=dict(passed=True,runs=len(checks),rollouts=total,checks=checks,scope='Exported records/actions/snapshots/media/CSV/file-integrity; no model execution and no checkpoint rehash')
 (R/'results/verification.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))
if __name__=='__main__':verify()
