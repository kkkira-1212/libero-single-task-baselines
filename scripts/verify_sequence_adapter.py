"""Real data boundary/time-index checks without importing any new policy dependencies."""
from pathlib import Path
import json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import h5py
import numpy as np
from adapters.libero_sequence import LiberoSequenceDataset
path=Path.home()/"scratch/robot_manipulation/datasets/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5"
split=json.loads(Path("configs/act/split.json").read_text())
results=[]
with h5py.File(path,"r") as f:
    for history in (2,8):
        ds=LiberoSequenceDataset(path,split["train"][:2],16,history)
        for index in (0,ds.ends[0]-1,ds.ends[0],len(ds)-1):
            sample=ds[index]
            assert sample["obs"]["agentview"].shape==(history,3,128,128)
            assert sample["obs"]["agent_pos"].shape==(history,9)
            assert sample["action"].shape==(16,7)
            raw=f["data"][sample["episode"]]
            np.testing.assert_array_equal(sample["action"][history-1],
                                          raw["actions"][sample["timestep"]].astype(np.float32))
            state=np.concatenate([raw["obs/joint_states"][sample["timestep"]],
                                  raw["obs/gripper_states"][sample["timestep"]]]).astype(np.float32)
            np.testing.assert_array_equal(sample["obs"]["agent_pos"][-1],state)
            assert sample["obs"]["agentview"].min()>=0 and sample["obs"]["agentview"].max()<=1
        results.append({"n_obs_steps":history,"boundary_checks":4,"passed":True})
Path("logs/sequence-adapter-verification.json").write_text(json.dumps(results,indent=2)+"\n")
print(json.dumps(results))
