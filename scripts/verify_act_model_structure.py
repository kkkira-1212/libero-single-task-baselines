"""Construct CPU model and check dimensions; no forward, backward, or weight download."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
torch.set_num_threads(1)
from adapters.act_model import build_act
model = build_act(pretrained_backbone=False)
assert model.input_proj_robot_state.in_features == 9
assert model.encoder_joint_proj.in_features == 9
assert model.encoder_action_proj.in_features == 7
assert model.action_head.out_features == 7
assert model.num_queries == 20
assert next(model.parameters()).device.type == "cpu"
print("PASS: author ACT constructed on CPU with state=9, action=7, chunk=20")
print("No forward/backward executed; no pretrained weights requested")
