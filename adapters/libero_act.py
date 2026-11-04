"""LIBERO HDF5 -> author ACT tensors, with episode isolation and shared preprocessing."""
import bisect
from pathlib import Path
import h5py
import numpy as np

CAMERAS = ("agentview_rgb", "eye_in_hand_rgb")
ONLINE_CAMERAS = ("agentview_image", "robot0_eye_in_hand_image")

def make_split(names, seed=0):
    names = sorted(names, key=lambda n: int(n.split("_")[-1]))
    order = np.random.RandomState(seed).permutation(len(names))
    n = int(len(names) * 0.8)
    if n == 0 or n == len(names):
        raise ValueError("Need nonempty train and validation sets")
    return {"seed": seed, "train": [names[i] for i in order[:n]],
            "validation": [names[i] for i in order[n:]]}

def read_state(obs, selection):
    return np.concatenate((obs["joint_states"][selection],
                           obs["gripper_states"][selection]), axis=-1).astype(np.float32)

def fit_stats(path, training_names):
    if not training_names or len(set(training_names)) != len(training_names):
        raise ValueError("Training names must be nonempty and unique")
    states, actions = [], []
    with h5py.File(path, "r") as f:
        for name in training_names:
            d = f["data"][name]
            states.append(read_state(d["obs"], slice(None)))
            actions.append(d["actions"][:].astype(np.float32))
    result = {"training_episodes": list(training_names), "std_floor": 0.01}
    for key, values in (("state", states), ("action", actions)):
        x = np.concatenate(values).astype(np.float64)
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite " + key)
        result[key + "_mean"] = x.mean(axis=0).tolist()
        result[key + "_std"] = np.maximum(x.std(axis=0, ddof=1), 0.01).tolist()
    return result

def normalize(value, stats, kind):
    return ((value - np.asarray(stats[kind + "_mean"], dtype=np.float32)) /
            np.asarray(stats[kind + "_std"], dtype=np.float32)).astype(np.float32)

def prepare_images(images):
    x = np.stack(images)
    if x.shape != (2, 128, 128, 3) or x.dtype != np.uint8:
        raise ValueError("Expected two uint8 128x128 RGB images")
    # Both stored LIBERO images and raw online observations use OpenGL convention.
    # Keep orientation identical; PNG display flips are not policy preprocessing.
    x = x.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[None, :, None, None]
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[None, :, None, None]
    return np.ascontiguousarray((x - mean) / std)

def online_observation(obs, stats):
    state = np.concatenate((obs["robot0_joint_pos"], obs["robot0_gripper_qpos"])).astype(np.float32)
    if state.shape != (9,):
        raise ValueError("Expected 9D state")
    return {"state": normalize(state, stats, "state"),
            "images": prepare_images([obs[k] for k in ONLINE_CAMERAS])}

def decode_action(predicted, stats, clip=True):
    action = (np.asarray(predicted, dtype=np.float32) *
              np.asarray(stats["action_std"], dtype=np.float32) +
              np.asarray(stats["action_mean"], dtype=np.float32))
    if action.shape[-1] != 7 or not np.isfinite(action).all():
        raise ValueError("Expected finite 7D action")
    return np.clip(action, -1, 1) if clip else action

class LiberoACTDataset:
    """One sample per real timestep; opens HDF5 per access for worker safety."""
    def __init__(self, path, episodes, stats, chunk_size=20, action_offset=0, require_next_action=False):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.path = str(Path(path).expanduser())
        self.episodes = list(episodes)
        if not self.episodes or len(set(self.episodes)) != len(self.episodes):
            raise ValueError("Episodes must be nonempty and unique")
        self.stats = stats
        self.chunk_size = chunk_size
        if action_offset not in (0,1) or (action_offset and not require_next_action):
            raise ValueError("Invalid explicit action offset/anchor contract")
        self.action_offset=action_offset
        self.require_next_action=require_next_action
        self.lengths = []
        with h5py.File(self.path, "r") as f:
            for name in self.episodes:
                d = f["data"][name]
                n = len(d["actions"])
                if n <= 0 or d["actions"].shape[1:] != (7,):
                    raise ValueError("Invalid episode " + name)
                if any(len(d["obs"][k]) != n for k in
                       CAMERAS + ("joint_states", "gripper_states")):
                    raise ValueError("Observation/action length mismatch")
                self.lengths.append(n)
        self.sample_lengths=[n-int(require_next_action) for n in self.lengths]
        if min(self.sample_lengths)<1:raise ValueError("No valid anchors")
        self.ends = np.cumsum(self.sample_lengths).tolist()

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        ep = bisect.bisect_right(self.ends, index)
        t = index - (self.ends[ep-1] if ep else 0)
        action_start=t+self.action_offset
        valid = min(self.chunk_size, self.lengths[ep] - action_start)
        with h5py.File(self.path, "r") as f:
            d = f["data"][self.episodes[ep]]
            state = read_state(d["obs"], t)
            images = prepare_images([d["obs"][k][t] for k in CAMERAS])
            raw = d["actions"][action_start:action_start+valid].astype(np.float32)
        actions = np.zeros((self.chunk_size, 7), dtype=np.float32)
        actions[:valid] = normalize(raw, self.stats, "action")
        is_pad = np.arange(self.chunk_size) >= valid
        return {"images": images, "state": normalize(state, self.stats, "state"),
                "actions": actions, "is_pad": is_pad,
                "episode": self.episodes[ep], "timestep": t}
