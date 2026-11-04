"""Prepare BAKU PKL with explicit demo split and existing language embedding."""
import argparse,json,pickle
from pathlib import Path
import h5py
import numpy as np
parser=argparse.ArgumentParser()
parser.add_argument("--embedding",required=True,help="Existing MiniLM task embedding .npy; no downloads here")
parser.add_argument("--output-root",required=True)
args=parser.parse_args()
embedding=np.load(args.embedding,allow_pickle=False).astype(np.float32)
assert embedding.shape==(384,) and np.isfinite(embedding).all()
split=json.loads(Path("configs/act/split.json").read_text())
path=Path.home()/"scratch/robot_manipulation/datasets/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5"
task="pick_up_the_alphabet_soup_and_place_it_in_the_basket"
with h5py.File(path,"r") as f:
    for subset in ("train","validation"):
        result={"observations":[],"states":[],"actions":[],"rewards":[],
                "task_emb":embedding,"episode_ids":split[subset],
                "provenance":{"timing":"retain stored post-action observations, unlike upstream rerender path",
                              "image_convention":"vertically flipped both offline and online",
                              "dataset_sha256":split["dataset_sha256"]}}
        for name in split[subset]:
            d=f["data"][name]
            result["observations"].append({
                "pixels":np.ascontiguousarray(d["obs/agentview_rgb"][:][:,::-1]),
                "pixels_egocentric":np.ascontiguousarray(d["obs/eye_in_hand_rgb"][:][:,::-1]),
                "joint_states":d["obs/joint_states"][:].astype(np.float32),
                "gripper_states":d["obs/gripper_states"][:].astype(np.float32),
                "robot_states":d["robot_states"][:].astype(np.float32)})
            for key in ("states","actions","rewards"):
                result[key].append(d[key][:].astype(np.float32))
        folder=Path(args.output_root).expanduser()/subset/"libero_object"
        folder.mkdir(parents=True,exist_ok=True)
        destination=folder/(task+".pkl")
        if destination.exists():
            raise FileExistsError(destination)
        with destination.open("wb") as target:
            pickle.dump(result,target,protocol=4)
        print(subset,len(result["observations"]),destination)
