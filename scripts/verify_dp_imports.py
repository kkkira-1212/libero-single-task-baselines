"""Import-only verification; no model construction, GPU, or simulation."""
import importlib.metadata as metadata
import json
from pathlib import Path
import sys
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
report={"passed":True,"python":sys.version,
        "classes":[c.__name__ for c in (DiffusionUnetImagePolicy,LinearNormalizer,
                                      MultiImageObsEncoder,DDPMScheduler)],
        "versions":{n:metadata.version(n) for n in ["torch","torchvision","diffusers","huggingface-hub","zarr","numcodecs"]},
        "model_operations_tested":False,"simulation_tested":False}
Path("logs/dp-import-check.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2),flush=True)
