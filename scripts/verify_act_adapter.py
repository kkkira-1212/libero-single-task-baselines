"""Read-only real-data contract checks, plus deterministic protocol files."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import h5py
import numpy as np
from adapters.libero_act import (LiberoACTDataset, make_split, fit_stats, read_state,
                                online_observation, normalize, decode_action)

project = Path(__file__).resolve().parents[1]
path = Path.home() / "scratch/robot_manipulation/datasets/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5"
with h5py.File(path, "r") as f:
    split = make_split(list(f["data"]), seed=0)
assert len(split["train"]) == 40 and len(split["validation"]) == 10
assert not (set(split["train"]) & set(split["validation"]))
assert split == make_split(split["train"] + split["validation"], seed=0)
stats = fit_stats(path, split["train"])
assert set(stats["training_episodes"]) == set(split["train"])
for kind, dim in [("state", 9), ("action", 7)]:
    assert len(stats[kind+"_mean"]) == dim
    assert np.isfinite(stats[kind+"_mean"]).all()
    assert (np.asarray(stats[kind+"_std"]) >= 0.01).all()

dataset = LiberoACTDataset(path, split["train"], stats, chunk_size=20)
checked = 0
with h5py.File(path, "r") as f:
    independent_states = np.concatenate(
        [read_state(f["data"][name]["obs"], slice(None)) for name in split["train"]])
    np.testing.assert_allclose(stats["state_mean"], independent_states.mean(0, dtype=np.float64))
    for ep, name in enumerate(dataset.episodes):
        first = dataset.ends[ep-1] if ep else 0
        last = dataset.ends[ep]-1
        for index in (first, last):
            sample = dataset[index]
            t = sample["timestep"]
            d = f["data"][name]
            assert sample["episode"] == name
            assert sample["state"].shape == (9,)
            assert sample["images"].shape == (2, 3, 128, 128)
            assert sample["actions"].shape == (20, 7)
            assert sample["is_pad"].dtype == np.bool_
            valid = min(20, len(d["actions"])-t)
            assert np.array_equal(sample["is_pad"], np.arange(20) >= valid)
            raw = d["actions"][t:t+valid].astype(np.float32)
            np.testing.assert_allclose(decode_action(sample["actions"][:valid], stats, clip=False),
                                       raw, atol=2e-7)
            assert np.all(sample["actions"][valid:] == 0)
            # Replay an offline record through the online preprocessing interface.
            online = online_observation({
                "robot0_joint_pos": d["obs/joint_states"][t],
                "robot0_gripper_qpos": d["obs/gripper_states"][t],
                "agentview_image": d["obs/agentview_rgb"][t],
                "robot0_eye_in_hand_image": d["obs/eye_in_hand_rgb"][t]}, stats)
            np.testing.assert_array_equal(online["images"], sample["images"])
            np.testing.assert_array_equal(online["state"], sample["state"])
            assert np.isfinite(sample["images"]).all()
            checked += 1
    # Crossing an episode index selects the next demo, never the previous action tail.
    for i in range(len(dataset.ends)-1):
        assert dataset[dataset.ends[i]]["episode"] == dataset.episodes[i+1]
    # Validation is readable with training statistics but never used to fit them.
    validation = LiberoACTDataset(path, split["validation"], stats, 20)
    assert validation[0]["state"].shape == (9,)
for index in (-1, len(dataset)):
    try:
        dataset[index]
        raise AssertionError("Invalid index accepted")
    except IndexError:
        pass
split["dataset_sha256"] = "42189d4415d4c51aaaf0708300653fccc39239cd3f2709079a713cd8d1678a8d"
for name, value in [("split.json", split), ("normalization.json", stats)]:
    dest = project / "configs/act" / name
    if dest.exists():
        assert json.loads(dest.read_text()) == value, "Protocol changed; inspect before overwriting"
    else:
        dest.write_text(json.dumps(value, indent=2)+"\n")
report = {"passed": True, "train_episodes":40, "validation_episodes":10,
          "train_timesteps":len(dataset), "validation_timesteps":len(validation),
          "boundary_samples_checked": checked, "chunk_size":20,
          "state_dim":9,"action_dim":7,
          "checks":["disjoint deterministic split","train-only normalization",
                    "episode boundary/padding","action normalization roundtrip",
                    "offline/online preprocessing on identical inputs"],
          "online_environment_parity_tested":False, "model_forward_backward_tested":False}
(project / "logs/act-adapter-verification.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
