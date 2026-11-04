"""One real-batch integration check; not a training experiment."""
import json
import os
from pathlib import Path
import socket
import sys
import time

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
assert os.environ.get("SLURM_JOB_ID"), "Use a compute allocation"
job = os.environ["SLURM_JOB_ID"]
out = root / "artifacts/act-batch" / job
out.mkdir(parents=True, exist_ok=True)
report = {"job_id": job, "host": socket.gethostname(), "passed": False,
          "purpose": "single_batch_integration_check_not_baseline_training"}
start = time.time()
try:
    import numpy as np
    import torch
    from adapters.libero_act import LiberoACTDataset
    from adapters.act_model import build_act, act_loss
    torch.set_num_threads(4)
    torch.manual_seed(0)
    np.random.seed(0)
    assert torch.cuda.is_available(), "CUDA unavailable"
    report.update(torch=torch.__version__, cuda_build=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(0))
    print("Imports and GPU ready", flush=True)
    split = json.loads((root / "configs/act/split.json").read_text())
    stats = json.loads((root / "configs/act/normalization.json").read_text())
    config = json.loads((root / "configs/act/interface.json").read_text())
    path = Path.home() / "scratch/robot_manipulation/datasets/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5"
    dataset = LiberoACTDataset(path, split["train"], stats, config["chunk_size"])
    # Full action chunk plus the final real step of the same episode.
    samples = [dataset[0], dataset[dataset.ends[0]-1]]
    batch = {key: torch.from_numpy(np.stack([s[key] for s in samples])).cuda()
             for key in ("images", "state", "actions", "is_pad")}
    report["batch_shapes"] = {k: list(v.shape) for k,v in batch.items()}
    report["valid_actions_per_sample"] = (~batch["is_pad"]).sum(1).cpu().tolist()
    model = build_act(chunk_size=config["chunk_size"],
                      state_dim=config["state_dim"], action_dim=config["action_dim"],
                      pretrained_backbone=False).cuda()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    print("ACT constructed; running real batch", flush=True)
    predicted, _, (mu, logvar) = model(
        batch["state"], batch["images"], None, batch["actions"], batch["is_pad"])
    assert tuple(predicted.shape) == (2,20,7), predicted.shape
    losses = act_loss(predicted, batch["actions"], batch["is_pad"], mu, logvar)
    assert all(torch.isfinite(value).item() for value in losses.values())
    changed = predicted.detach().clone()
    changed[batch["is_pad"]] += 100
    masked = act_loss(changed, batch["actions"], batch["is_pad"], mu.detach(), logvar.detach())
    torch.testing.assert_close(masked["l1"], losses["l1"].detach())
    report["padding_loss_invariance_passed"] = True
    losses["loss"].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all().item() for g in grads)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    assert torch.isfinite(grad_norm).item() and grad_norm.item() > 0
    optimizer.step()
    torch.cuda.synchronize()
    report["losses"] = {k: float(v.detach().cpu()) for k,v in losses.items()}
    report["gradient_norm_before_clip"] = float(grad_norm)
    report["optimizer_steps"] = 1
    print("Forward, backward and one optimizer step passed", flush=True)
    model.eval()
    with torch.no_grad():
        before = model(batch["state"], batch["images"], None)[0].clone()
    assert torch.isfinite(before).all()
    checkpoint_dir = Path.home() / "scratch/robot_manipulation/act-batch" / job
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / "interface-check.pt"
    torch.save({"model": {k:v.detach().cpu() for k,v in model.state_dict().items()},
                "interface": config, "split":split, "normalization": stats,
                "source_commit":"742c753c0d4a5d87076c8f69e5628c79a8cc5488",
                "purpose":report["purpose"],
                "optimizer_state_saved":False}, str(checkpoint))
    # Perturb, then restore, to exercise load_state_dict rather than just reread bytes.
    with torch.no_grad():
        model.action_head.bias.add_(1)
    saved = torch.load(str(checkpoint), map_location="cpu")
    model.load_state_dict(saved["model"], strict=True)
    with torch.no_grad():
        after = model(batch["state"], batch["images"], None)[0]
    torch.cuda.synchronize()
    torch.testing.assert_close(before, after, rtol=1e-5, atol=1e-6)
    report.update(checkpoint=str(checkpoint), checkpoint_bytes=checkpoint.stat().st_size,
                  checkpoint_reload_passed=True,
                  reload_prediction_max_error=float((before-after).abs().max().item()),
                  optimizer_resume_tested=False, online_rollout_tested=False,
                  max_gpu_memory_bytes=torch.cuda.max_memory_allocated(), passed=True)
except Exception as exc:
    report["error"] = repr(exc)
    import traceback
    traceback.print_exc()
finally:
    report["wall_seconds"] = time.time()-start
    (out/"result.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2),flush=True)
sys.exit(0 if report["passed"] else 1)
