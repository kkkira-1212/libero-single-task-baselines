"""Causal LIBERO labels using the unchanged author BCDataset sampler.

The sampler already excludes the last observation for history=False.
Shift each episode's action array left by1 and append one zero command.
Thus action slot[t] becomes original command[t+1]; trailing zero-padding and
episode-uniform sampling remain the author's conventions. PKLs are unchanged.
"""
import numpy as np
from read_data.libero import BCDataset

class CausalBCDataset(BCDataset):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self._obs_type!="pixels" or self._prompt!="text" or self._history_len!=1:
            raise ValueError("Validated causal BAKU adapter requires pixels/text/history=False")
        for episodes in self._episodes.values():
            for episode in episodes:
                old=episode["action"]
                if len(old)<2 or old.shape[1:]!=(7,):
                    raise ValueError("Invalid causal episode")
                episode["action"]=np.concatenate([old[1:].copy(),np.zeros_like(old[:1])],axis=0)
