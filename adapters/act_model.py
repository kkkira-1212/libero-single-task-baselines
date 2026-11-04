"""Author ACT builder without global CLI parsing, implicit CUDA or weight downloads."""
from pathlib import Path
import sys
from types import SimpleNamespace

def build_act(chunk_size=20, state_dim=9, action_dim=7, pretrained_backbone=False):
    root = Path(__file__).resolve().parents[1] / "third_party/ACT/detr"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from models.detr_vae import build
    args = SimpleNamespace(
        state_dim=state_dim, action_dim=action_dim,
        num_queries=chunk_size, camera_names=["agentview", "eye_in_hand"],
        backbone="resnet18", lr_backbone=1e-5, masks=False, dilation=False,
        position_embedding="sine", hidden_dim=512, nheads=8,
        enc_layers=4, dec_layers=7, dim_feedforward=3200,
        dropout=0.1, pre_norm=False, pretrained_backbone=pretrained_backbone)
    return build(args)

def act_loss(predicted, target, is_pad, mu, logvar, kl_weight=10.0):
    """Matches author's padded-mean L1 + KL convention; not valid-token renormalized."""
    import torch
    l1 = (torch.abs(predicted - target) * (~is_pad).unsqueeze(-1)).mean()
    kl = (-0.5 * (1 + logvar - mu.square() - logvar.exp())).sum(-1).mean()
    return {"loss": l1 + kl_weight * kl, "l1": l1, "kl": kl}
