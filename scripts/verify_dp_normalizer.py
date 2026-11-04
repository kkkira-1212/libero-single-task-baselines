"""CPU-only regression: affine math, nested storage, strict reload and real data."""
from pathlib import Path
import json
import io
import torch
import numpy as np
from diffusion_policy.model.common.normalizer import LinearNormalizer,SingleFieldLinearNormalizer
from adapters.dp_model import make_normalizer

torch.set_num_threads(1)
x=torch.tensor([[-2.,1.,3.],[0.,2.,3.],[2.,3.,3.]])
normalizer=LinearNormalizer()
normalizer.fit({"action":x},mode="limits")
expected=torch.tensor([[-1.,-1.,0.],[0.,0.,0.],[1.,1.,0.]])
torch.testing.assert_close(normalizer["action"].normalize(x),expected)
torch.testing.assert_close(normalizer["action"].unnormalize(expected),x)
normalizer["image"]=SingleFieldLinearNormalizer.create_identity()
images=torch.rand(2,3,8,8)
torch.testing.assert_close(normalizer["image"].normalize(images),images)
assert all(not p.requires_grad for p in normalizer.parameters())
buffer=io.BytesIO()
torch.save(normalizer.state_dict(),buffer)
buffer.seek(0)
restored=LinearNormalizer()
restored.load_state_dict(torch.load(buffer,map_location="cpu"),strict=True)
torch.testing.assert_close(restored["action"].normalize(x),expected)
torch.testing.assert_close(restored["image"].normalize(images),images)
assert set(restored.state_dict())==set(normalizer.state_dict())
assert all(not p.requires_grad for p in restored.parameters())
split=json.loads(Path("configs/act/split.json").read_text())
data=Path.home()/"scratch/robot_manipulation/datasets/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5"
real=make_normalizer(data,split["train"])
sample=torch.tensor([[0.,0.,0.,0.,0.,0.,-1.],[.1,-.2,.3,.04,-.05,.06,1.]])
torch.testing.assert_close(real["action"].unnormalize(real["action"].normalize(sample)),sample,rtol=1e-5,atol=1e-6)
copy=LinearNormalizer()
copy.load_state_dict(real.state_dict(),strict=True)
torch.testing.assert_close(copy["action"].normalize(sample),real["action"].normalize(sample))
report={"passed":True,"device":"cpu","torch":torch.__version__,
        "checks":["analytic_limits_with_constant_channel","inverse_roundtrip",
                  "image_identity","nontrainable_stats","nested_strict_state_reload",
                  "real_training_only_fit_and_reload"],
        "gpu_model_tested":False}
Path("logs/dp-normalizer-compat-check.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2),flush=True)
