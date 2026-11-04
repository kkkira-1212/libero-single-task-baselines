"""Raw temporal samples for diffusion/A2A adapters; no model-specific normalization."""
from pathlib import Path
import bisect
import h5py
import numpy as np
from adapters.libero_causal_contract import sequence_indices

class LiberoSequenceDataset:
    def __init__(self,path,episodes,horizon=16,n_obs_steps=2,action_offset=0,require_next_action=False):
        if not 1<=n_obs_steps<=horizon:
            raise ValueError("Invalid horizon/history")
        self.path=str(Path(path).expanduser())
        self.episodes=list(episodes)
        self.horizon=horizon
        self.n_obs_steps=n_obs_steps
        self.action_offset=action_offset
        self.require_next_action=require_next_action
        if not self.episodes or len(set(self.episodes))!=len(self.episodes):
            raise ValueError("Episodes must be nonempty and unique")
        with h5py.File(self.path,"r") as f:
            self.lengths=[len(f["data"][n]["actions"]) for n in self.episodes]
        self.sample_lengths=[n-int(require_next_action) for n in self.lengths]
        if min(self.sample_lengths)<1: raise ValueError("No valid sample anchors")
        sequence_indices(self.lengths[0],0,horizon,n_obs_steps,action_offset,require_next_action)
        self.ends=np.cumsum(self.sample_lengths).tolist()
    def __len__(self):
        return self.ends[-1]
    def __getitem__(self,index):
        if not 0<=index<len(self):
            raise IndexError(index)
        ep=bisect.bisect_right(self.ends,index)
        t=index-(self.ends[ep-1] if ep else 0)
        idx=sequence_indices(self.lengths[ep],t,self.horizon,self.n_obs_steps,
                             self.action_offset,self.require_next_action)
        clipped=idx["action_indices"]
        with h5py.File(self.path,"r") as f:
            d=f["data"][self.episodes[ep]]
            # Scalar reads support repeated boundary indices without h5py fancy-index restrictions.
            action=np.stack([d["actions"][int(i)] for i in clipped]).astype(np.float32)
            obs_ids=idx["obs_indices"]
            obs={}
            for target,source in [("agentview","agentview_rgb"),("wrist","eye_in_hand_rgb")]:
                image=np.stack([d["obs"][source][int(i)] for i in obs_ids])
                obs[target]=np.ascontiguousarray(image.transpose(0,3,1,2).astype(np.float32)/255.)
            obs["agent_pos"]=np.stack([
                np.concatenate([d["obs/joint_states"][int(i)],d["obs/gripper_states"][int(i)]])
                for i in obs_ids]).astype(np.float32)
        return {"obs":obs,"action":action,"episode":self.episodes[ep],"timestep":t,
                "padding":idx["action_padding"],"obs_padding":idx["obs_padding"],
                "current_action_index":self.n_obs_steps-1,"action_offset":self.action_offset}
