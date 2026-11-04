"""GPU arithmetic only; no simulation, model, training, or installation."""
import json
import os
from pathlib import Path
import socket
import sys
import time

assert os.environ.get("SLURM_JOB_ID"), "Use a compute allocation."
out=Path("artifacts/cuda")/os.environ["SLURM_JOB_ID"]
out.mkdir(parents=True,exist_ok=True)
start=time.time()
report={"job_id":os.environ["SLURM_JOB_ID"],"host":socket.gethostname(),
        "cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),"passed":False}
try:
    import torch
    report["torch"]=torch.__version__
    report["torch_cuda_build"]=torch.version.cuda
    report["cuda_available"]=torch.cuda.is_available()
    assert report["cuda_available"], "CUDA unavailable"
    report["gpu"]=torch.cuda.get_device_name(0)
    report["capability"]=list(torch.cuda.get_device_capability(0))
    report["torch_arch_list"]=torch.cuda.get_arch_list()
    x=torch.tensor([1.0,2.0,3.0],device="cuda")
    y=x+1
    torch.cuda.synchronize()
    report["result"]=y.cpu().tolist()
    assert report["result"]==[2.0,3.0,4.0]
    report["passed"]=True
except Exception as exc:
    report["error"]=repr(exc)
finally:
    report["wall_seconds"]=time.time()-start
    (out/"result.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2),flush=True)
sys.exit(0 if report["passed"] else 1)
