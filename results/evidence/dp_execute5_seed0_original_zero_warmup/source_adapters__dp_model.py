"""Build author Diffusion Policy and train-only normalization for LIBERO."""
import h5py
import numpy as np

def make_normalizer(dataset_path, training_episodes):
    from diffusion_policy.model.common.normalizer import LinearNormalizer,SingleFieldLinearNormalizer
    states=[]; actions=[]
    with h5py.File(dataset_path,"r") as f:
        for name in training_episodes:
            d=f["data"][name]
            states.append(np.concatenate([d["obs/joint_states"][:],d["obs/gripper_states"][:]],axis=-1))
            actions.append(d["actions"][:])
    normalizer=LinearNormalizer()
    normalizer.fit({"agent_pos":np.concatenate(states).astype(np.float32),
                    "action":np.concatenate(actions).astype(np.float32)},
                   last_n_dims=1,mode="limits")
    for key in ("agentview","wrist"):
        normalizer[key]=SingleFieldLinearNormalizer.create_identity()
    return normalizer

def build_dp(pretrained=False,inference_steps=10):
    import torch
    from torchvision.models import resnet18
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
    from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
    shape={"action":{"shape":[7]},
           "obs":{"agentview":{"shape":[3,128,128],"type":"rgb"},
                  "wrist":{"shape":[3,128,128],"type":"rgb"},
                  "agent_pos":{"shape":[9],"type":"low_dim"}}}
    rgb=resnet18(pretrained=pretrained)
    rgb.fc=torch.nn.Identity()
    encoder=MultiImageObsEncoder(shape_meta=shape,rgb_model=rgb,
                                crop_shape=None,use_group_norm=True,
                                share_rgb_model=False,imagenet_norm=True)
    scheduler=DDPMScheduler(num_train_timesteps=100,beta_start=0.0001,
                           beta_end=0.02,beta_schedule="squaredcos_cap_v2",
                           variance_type="fixed_small",clip_sample=True,prediction_type="epsilon")
    return DiffusionUnetImagePolicy(shape_meta=shape,noise_scheduler=scheduler,
                                    obs_encoder=encoder,horizon=16,n_obs_steps=2,n_action_steps=8,
                                    num_inference_steps=inference_steps,obs_as_global_cond=True,
                                    diffusion_step_embed_dim=256,down_dims=(256,512,1024),
                                    kernel_size=5,n_groups=8,cond_predict_scale=True)

def load_dp_state(policy, state):
    """Nested normalizer load rebuilds tensors; restore the policy's target placement."""
    device, dtype = policy.device, policy.dtype
    policy.load_state_dict(state, strict=True)
    return policy.to(device=device, dtype=dtype)

def assert_dp_placement(policy, device):
    """Fail before an expensive forward if dynamically loaded statistics stayed on CPU."""
    for name,value in list(policy.named_parameters()) + list(policy.named_buffers()):
        if value.device != device:
            raise RuntimeError("Device mismatch at %s: %s vs %s" % (name,value.device,device))
