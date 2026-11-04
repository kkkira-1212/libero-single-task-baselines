"""Read-only streamed checkpoint/dataset integrity check on a Slurm CPU node."""
import hashlib,json,os,time
from pathlib import Path
R=Path(__file__).resolve().parents[1]
assert os.environ.get('SLURM_JOB_ID'),'Use a Slurm compute node'
start=time.time();files={}
runs=json.loads((R/'results/runs.json').read_text())['runs']
for row in runs:
 path=row['checkpoint'];expected=row['checkpoint_sha256']
 if path in files:assert files[path]['expected']==expected
 else:files[path]=dict(expected=expected)
p=Path.home()/'scratch/robot_manipulation/datasets/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5'
files[str(p)]=dict(expected=json.loads((R/'configs/act/split.json').read_text())['dataset_sha256'])
for path,item in files.items():
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 item.update(actual=h.hexdigest(),bytes=Path(path).stat().st_size,passed=h.hexdigest()==item['expected'])
 print(Path(path).parent.name,item['passed'],flush=True)
report=dict(job=os.environ['SLURM_JOB_ID'],passed=all(x['passed'] for x in files.values()),files=files,wall_seconds=time.time()-start)
out=R/'artifacts/release-assets'/os.environ['SLURM_JOB_ID'];out.mkdir(parents=True,exist_ok=True)
(out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
assert report['passed']
