"""Fit eight fixed training examples as a diagnostic, never a benchmark result."""
import csv
import json
import os
from pathlib import Path
import socket
import sys
import time

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
assert os.environ.get("SLURM_JOB_ID"), "Use Slurm compute"
job=os.environ["SLURM_JOB_ID"]
out=root/"artifacts/act-tiny"/job
out.mkdir(parents=True,exist_ok=True)
start=time.time()
report={"job_id":job,"host":socket.gethostname(),
        "purpose":"tiny_training_subset_diagnostic_only","execution_ok":False,
        "target_updates":100,"seed":0,"pretrained_backbone":False,
        "online_rollouts":0}
history=[]
try:
    import numpy as np
    import torch
    from adapters.libero_act import LiberoACTDataset
    from adapters.act_model import build_act,act_loss
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(4)
    assert torch.cuda.is_available()
    report.update(torch=torch.__version__,cuda_build=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(0))
    split=json.loads((root/"configs/act/split.json").read_text())
    stats=json.loads((root/"configs/act/normalization.json").read_text())
    config=json.loads((root/"configs/act/interface.json").read_text())
    names=split["train"][:2]
    assert not (set(names)&set(split["validation"]))
    path=Path.home()/"scratch/robot_manipulation/datasets/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5"
    ds=LiberoACTDataset(path,names,stats,20)
    samples=[]
    for ep,n in enumerate(ds.lengths):
        offset=ds.ends[ep-1] if ep else 0
        for t in [0,n//3,2*n//3,n-1]:
            samples.append(ds[offset+t])
    report["samples"]=[{"episode":s["episode"],"timestep":s["timestep"]} for s in samples]
    batch={k:torch.from_numpy(np.stack([s[k] for s in samples])).cuda()
           for k in ("images","state","actions","is_pad")}
    model=build_act(pretrained_backbone=False).cuda()
    optimizer=torch.optim.AdamW([
        {"params":[p for name,p in model.named_parameters() if "backbone" not in name],"lr":5e-5},
        {"params":[p for name,p in model.named_parameters() if "backbone" in name],"lr":1e-5}
    ],weight_decay=1e-4)
    report["hyperparameters"]={"batch_size":4,"lr":5e-5,"backbone_lr":1e-5,
                              "kl_weight":10,"grad_clip_norm":1,"chunk_size":20}
    def evaluate(update):
        model.eval()
        with torch.no_grad():
            predicted=model(batch["state"],batch["images"],None)[0]
            mask=(~batch["is_pad"]).unsqueeze(-1).expand_as(predicted)
            absolute=(predicted-batch["actions"]).abs()
            l1=(absolute*mask).sum()/mask.sum()
            per_dim=(absolute*mask).sum((0,1))/mask.sum((0,1))
        model.train()
        row={"update":update,"prior_inference_valid_l1":float(l1),
             "per_dim_l1":per_dim.cpu().tolist(),"elapsed_seconds":time.time()-start}
        assert np.isfinite(row["prior_inference_valid_l1"])
        return row
    initial=evaluate(0)
    history.append(initial)
    print("Initial:",json.dumps(initial),flush=True)
    norm_history=[]
    completed=0
    for step in range(1,101):
        # Reserve five minutes for checkpoint IO/cleanup within the 15-minute allocation.
        if time.time()-start>600:
            report["stopped_for_time_budget"]=True
            break
        indices=torch.randperm(len(samples),device="cuda")[:4]
        b={k:v[indices] for k,v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        pred,_,(mu,logvar)=model(b["state"],b["images"],None,b["actions"],b["is_pad"])
        losses=act_loss(pred,b["actions"],b["is_pad"],mu,logvar,10)
        assert torch.isfinite(losses["loss"]).item()
        losses["loss"].backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        assert torch.isfinite(norm).item()
        norm_history.append(float(norm))
        optimizer.step()
        completed=step
        if step%10==0:
            row=evaluate(step)
            row.update({k:float(v.detach()) for k,v in losses.items()})
            history.append(row)
            print(json.dumps(row),flush=True)
    final=evaluate(completed)
    report.update(initial_prior_l1=initial["prior_inference_valid_l1"],
                  final_prior_l1=final["prior_inference_valid_l1"],
                  final_per_dim_l1=final["per_dim_l1"],updates_completed=completed,
                  inference_error_decreased=final["prior_inference_valid_l1"]<initial["prior_inference_valid_l1"],
                  completed_target=completed==100,
                  grad_norm_max=max(norm_history) if norm_history else None,
                  gradient_clipped_fraction=float(np.mean(np.asarray(norm_history)>1)) if norm_history else None)
    history.append(final)
    checkpoint_dir=Path.home()/"scratch/robot_manipulation/act-tiny"/job
    checkpoint_dir.mkdir(parents=True,exist_ok=True)
    checkpoint=checkpoint_dir/"diagnostic.pt"
    torch.save({"model":{k:v.detach().cpu() for k,v in model.state_dict().items()},
                "interface":config,"normalization":stats,"split":split,
                "updates":completed,"purpose":report["purpose"],"samples":report["samples"],
                "source_commit":"742c753c0d4a5d87076c8f69e5628c79a8cc5488",
                "optimizer_state_saved":False},str(checkpoint))
    report.update(checkpoint=str(checkpoint),checkpoint_bytes=checkpoint.stat().st_size,
                  max_gpu_memory_bytes=torch.cuda.max_memory_allocated(),
                  execution_ok=True,optimizer_resume_tested=False)
except Exception as exc:
    report["error"]=repr(exc)
    import traceback
    traceback.print_exc()
finally:
    report["wall_seconds"]=time.time()-start
    (out/"history.json").write_text(json.dumps(history,indent=2)+"\n")
    if history:
        with (out/"learning_curve.csv").open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=["update","prior_inference_valid_l1","elapsed_seconds"],
                                  extrasaction="ignore")
            writer.writeheader()
            writer.writerows(history)
    (out/"result.json").write_text(json.dumps(report,indent=2)+"\n")
    print("Result:",json.dumps(report,indent=2),flush=True)
sys.exit(0 if report["execution_ok"] else 1)
