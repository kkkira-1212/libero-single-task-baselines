"""Generate local LIBERO paths and materialize archived integration reports."""
import json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def prepare():
 for folder in ['logs','artifacts','configs/libero']:(ROOT/folder).mkdir(parents=True,exist_ok=True)
 base=ROOT/'third_party/LIBERO/libero/libero'
 if not base.is_dir():raise FileNotFoundError('Bootstrap author repositories first')
 config=dict(benchmark_root=str(base),bddl_files=str(base/'bddl_files'),init_states=str(base/'init_files'),assets=str(base/'assets'),datasets=str(Path.home()/'scratch/robot_manipulation/datasets'))
 (ROOT/'configs/libero/config.yaml').write_text(json.dumps(config,indent=2)+'\n')
 for source in (ROOT/'results/gates').rglob('result.json'):
  target=ROOT/'artifacts'/source.relative_to(ROOT/'results/gates')
  if target.exists():
   if target.read_bytes()!=source.read_bytes():raise RuntimeError('Conflicting archived gate: '+str(target))
  else:
   target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
 print('Local LIBERO configuration and archived gate reports ready')
if __name__=='__main__':prepare()
