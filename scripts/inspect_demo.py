"""Read-only HDF5 schema inspection; no simulation, training, or full image loading."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("dataset")
parser.add_argument("--output", required=True)
args = parser.parse_args()
report = {"file": str(Path(args.dataset).resolve()), "episodes": []}
def attrs(group):
    return {k: str(v) for k, v in group.attrs.items()}
with h5py.File(args.dataset, "r") as f:
    report["root_attrs"] = attrs(f)
    group = f["data"]
    report["data_attrs"] = attrs(group)
    for name in sorted(group):
        episode = group[name]
        if not isinstance(episode, h5py.Group):
            continue
        item = {"name": name, "attrs": attrs(episode), "datasets": {}}
        def visit(key, value):
            if isinstance(value, h5py.Dataset):
                item["datasets"][key] = {
                    "shape": list(value.shape), "dtype": str(value.dtype),
                    "compression": value.compression}
        episode.visititems(visit)
        if "actions" in episode:
            actions = episode["actions"][()]
            item["action_summary"] = {
                "finite": bool(np.isfinite(actions).all()),
                "min": actions.min(axis=0).tolist(),
                "max": actions.max(axis=0).tolist()}
            length = len(actions)
            for key, value in item["datasets"].items():
                if key.startswith("obs/") and value["shape"]:
                    assert value["shape"][0] == length, (name, key)
        report["episodes"].append(item)
    report["episode_count"] = len(report["episodes"])
    report["mask_keys"] = list(f["mask"].keys()) if "mask" in f else []
Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
print("Episodes:", report["episode_count"])
print("Report:", args.output)
