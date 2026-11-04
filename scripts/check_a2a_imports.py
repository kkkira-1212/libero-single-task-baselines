"""CPU import and normalizer gate; no model training on login nodes."""
import json,os,sys
from pathlib import Path
import torch,numpy as np
import importlib.util
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from adapters.a2a_model import make_normalizer
from roboverse_learn.il.policies.a2a.a2a_policy import A2AImagePolicy
from roboverse_learn.il.utils.flow.flow_matchers import ConditionalFlowMatcher
import libero
from libero.libero.envs import OffScreenRenderEnv
assert any(str(ROOT/"third_party/LIBERO/libero")==str(Path(p).resolve()) for p in libero.__path__),list(libero.__path__)
assert os.environ.get("SLURM_JOB_ID")
split=json.loads((ROOT/"configs/act/split.json").read_text())
path=Path.home()/"scratch/robot_manipulation/datasets/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5"
normalizer=make_normalizer(path,split["train"])
x=torch.linspace(-0.5,0.5,14).reshape(2,7)
torch.testing.assert_close(normalizer["action"].unnormalize(normalizer["action"].normalize(x)),x,rtol=1e-5,atol=1e-6)
from roboverse_learn.il.utils.normalizer import LinearNormalizer
restored=LinearNormalizer();restored.load_state_dict(normalizer.state_dict())
torch.testing.assert_close(normalizer["action"].normalize(x),restored["action"].normalize(x))
assert all(not p.requires_grad for p in normalizer.parameters())
assert all(not p.requires_grad for p in restored.parameters())
for key in ("agentview","wrist"):
    image=torch.rand(2,3,8,8)
    torch.testing.assert_close(normalizer[key].normalize(image),image)
    torch.testing.assert_close(restored[key].normalize(image),image)
report={"passed":True,"torch":torch.__version__,"normalizer_train_only_roundtrip_reload":True,
        "imports":["A2AImagePolicy","ConditionalFlowMatcher","OffScreenRenderEnv"],
        "note":"CPU gate only;GPU model/live checks pending"}
(ROOT/"logs"/("a2a-imports-"+os.environ["SLURM_JOB_ID"]+".json")).write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2),flush=True)
