"""Construct author A2A core with LIBERO metadata; no RoboVerse simulator import."""
from pathlib import Path
import h5py,numpy as np

def make_normalizer(path,episodes):
    from roboverse_learn.il.utils.normalizer import LinearNormalizer,SingleFieldLinearNormalizer
    states=[];actions=[]
    with h5py.File(path,"r") as f:
        for name in episodes:
            d=f["data"][name]
            states.append(np.concatenate([d["obs/joint_states"][:],d["obs/gripper_states"][:]],axis=-1))
            actions.append(d["actions"][:])
    normalizer=LinearNormalizer()
    normalizer.fit({"agent_pos":np.concatenate(states).astype(np.float32),
                    "action":np.concatenate(actions).astype(np.float32)},last_n_dims=1,mode="limits")
    for key in ("agentview","wrist"):normalizer[key]=SingleFieldLinearNormalizer.create_identity()
    normalizer.requires_grad_(False)
    return normalizer

def build_a2a(sampling_steps=6):
    import torch
    from omegaconf import OmegaConf
    from torchvision.models import resnet18
    from roboverse_learn.il.policies.a2a.a2a_policy import A2AImagePolicy
    from roboverse_learn.il.utils.vision.multi_image_obs_encoder import MultiImageObsEncoder
    from roboverse_learn.il.utils.flow.flow_matchers import ConditionalFlowMatcher
    root=Path(__file__).resolve().parents[1]
    config=OmegaConf.load(root/"third_party/A2A/roboverse_learn/il/configs/policy_config/a2a.yaml")
    shape={"action":{"shape":[7]},"obs":{"agentview":{"shape":[3,128,128],"type":"rgb"},
           "wrist":{"shape":[3,128,128],"type":"rgb"},"agent_pos":{"shape":[9],"type":"low_dim"}}}
    rgb=resnet18(weights=None);rgb.fc=torch.nn.Identity()
    encoder=MultiImageObsEncoder(shape_meta=shape,rgb_model=rgb,crop_shape=None,
                  use_group_norm=True,share_rgb_model=False,imagenet_norm=True)
    return A2AImagePolicy(shape_meta=shape,obs_encoder=encoder,horizon=16,n_obs_steps=8,n_action_steps=8,
          flow_net=config.flow_net,flow_matcher=ConditionalFlowMatcher(sigma=0.0,num_sampling_steps=sampling_steps),
          decode_flow_latents=config.decode_flow_latents,consistency_weight=config.consistency_weight,
          enc_contrastive_weight=config.enc_contrastive_weight,flow_contrastive_weight=config.flow_contrastive_weight,
          latent_dim=config.latent_dim,action_ae=config.action_ae)

def load_state(policy,state):
    device,dtype=policy.device,policy.dtype
    policy.load_state_dict(state,strict=True);return policy.to(device=device,dtype=dtype)
def assert_placement(policy,device):
    assert all(not p.requires_grad for p in policy.normalizer.parameters()),"Normalizer statistics must be frozen"
    for name,value in list(policy.named_parameters())+list(policy.named_buffers()):
        if value.device!=device:raise RuntimeError("Device mismatch "+name)
