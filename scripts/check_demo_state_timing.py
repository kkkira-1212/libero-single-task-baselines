"""Compare stored joint observations to pre/next recorded simulator qpos."""
import json
from pathlib import Path
import h5py
import numpy as np

path=Path.home()/"scratch/robot_manipulation/datasets/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5"
split=json.loads(Path("configs/act/split.json").read_text())
rows=[]
with h5py.File(path,"r") as f:
    for name in split["train"][:2]:
        d=f["data"][name]
        states=d["states"][:]
        observed=np.concatenate([d["obs/joint_states"][:],d["obs/gripper_states"][:]],axis=-1)
        # MuJoCo flattened states begin with time, then qpos. Panda arm+gripper are first nine qpos.
        same=states[:-1,1:10]
        next_state=states[1:,1:10]
        target=observed[:-1]
        rows.append({"episode":name,
                     "same_state_joint_mae":float(np.abs(target[:,:7]-same[:,:7]).mean()),
                     "next_state_joint_mae":float(np.abs(target[:,:7]-next_state[:,:7]).mean()),
                     "same_state_gripper_mae":float(np.abs(target[:,7:]-same[:,7:]).mean()),
                     "next_state_gripper_mae":float(np.abs(target[:,7:]-next_state[:,7:]).mean()),
                     "first_obs":observed[0].tolist(),
                     "first_state_qpos":states[0,1:10].tolist(),
                     "next_state_qpos":states[1,1:10].tolist()})
Path("logs/demo-state-timing.json").write_text(json.dumps(rows,indent=2)+"\n")
print(json.dumps(rows,indent=2))
