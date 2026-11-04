"""Fetch exact author commits and apply the recorded patches. No pip installation."""
import argparse,json,subprocess,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--check-only',action='store_true');a=p.parse_args()
def run(argv,**kw):return subprocess.run(argv,check=True,**kw)
for item in json.loads((ROOT/'upstream/manifest.json').read_text())['repositories']:
 dest=ROOT/'third_party'/item['directory']
 if not dest.exists():
  if a.check_only:raise FileNotFoundError(dest)
  dest.parent.mkdir(parents=True,exist_ok=True)
  run(['git','clone','--filter=blob:none',item['url'],str(dest)])
  run(['git','-C',str(dest),'checkout','--detach',item['commit']])
 actual=subprocess.check_output(['git','-C',str(dest),'rev-parse','HEAD']).decode().strip()
 if actual!=item['commit']:raise RuntimeError('Unexpected commit: '+str(dest))
 expected=(ROOT/item['patch']).read_bytes() if item['patch'] else b''
 actual_diff=subprocess.check_output(['git','-C',str(dest),'diff','--binary','HEAD'])
 if actual_diff!=expected:
  if actual_diff or a.check_only:raise RuntimeError('Unexpected source modifications: '+str(dest))
  run(['git','-C',str(dest),'apply','--check',str(ROOT/item['patch'])])
  run(['git','-C',str(dest),'apply',str(ROOT/item['patch'])])
  assert subprocess.check_output(['git','-C',str(dest),'diff','--binary','HEAD'])==expected
 print(item['directory'],actual,'patch verified')
