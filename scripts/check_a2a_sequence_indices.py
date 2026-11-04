"""Compare LIBERO temporal indexing against actual upstream A2A sampler, without models."""
import ast,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
source=(ROOT/"third_party/A2A/roboverse_learn/il/utils/sampler.py").read_text()
tree=ast.parse(source);fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=="create_indices")
fn.decorator_list=[];module=ast.Module(body=[fn],type_ignores=[]);ast.fix_missing_locations(module)
scope={"np":np};exec(compile(module,"upstream_create_indices","exec"),scope)
lengths=[20,15];ends=np.cumsum(lengths)
indices=scope["create_indices"](ends,16,np.ones(2,dtype=bool),7,7)
checked=0;currents={0:set(),1:set()}
for bs,be,ss,se in indices:
 ep=0 if bs<ends[0] else 1;base=0 if ep==0 else int(ends[0]);length=lengths[ep]
 sequence=np.empty(16,dtype=int);sequence[ss:se]=np.arange(bs,be)
 if ss>0:sequence[:ss]=sequence[ss]
 if se<16:sequence[se:]=sequence[se-1]
 t=int(bs-ss+7-base)
 expected=np.clip(np.arange(t-7,t+9),0,length-1)+base
 np.testing.assert_array_equal(sequence,expected)
 np.testing.assert_array_equal(sequence[7:15],expected[7:15])
 currents[ep].add(t);checked+=1
assert all(currents[i]==set(range(n-1)) for i,n in enumerate(lengths))
report={"passed":True,"upstream_windows_compared":checked,"history_indices":[0,8],"future_action_indices":[7,15],
        "all_common_windows_match":True,"our_extra_current_timestep_per_episode":"last timestep L-1;upstream horizon16/pad7 ends at L-2",
        "note":"This validates sequence slicing only,not raw observation/action causal timestamps."}
out=ROOT/"artifacts/a2a-audit/static";out.mkdir(parents=True,exist_ok=True)
(out/"sequence_indices.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
