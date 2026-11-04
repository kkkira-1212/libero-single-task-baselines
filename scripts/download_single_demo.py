"""Download exactly one official demonstration file, pinned by Hub revision."""
import hashlib
import json
import os
from pathlib import Path
import urllib.request

repo = "yifengzhu-hf/LIBERO-datasets"
name = "libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5"
expected = "42189d4415d4c51aaaf0708300653fccc39239cd3f2709079a713cd8d1678a8d"
expected_size = 780145352
with urllib.request.urlopen("https://huggingface.co/api/datasets/" + repo, timeout=60) as response:
    revision = json.load(response)["sha"]
dest = Path.home() / "scratch/robot_manipulation/datasets" / name
dest.parent.mkdir(parents=True, exist_ok=True)
url = "https://huggingface.co/datasets/" + repo + "/resolve/" + revision + "/" + name
if not dest.exists():
    partial = dest.with_suffix(".hdf5.partial")
    print("Downloading", url, flush=True)
    with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as output:
        while True:
            chunk = response.read(4 * 1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
    target = partial
else:
    target = dest
digest = hashlib.sha256()
with target.open("rb") as f:
    for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
        digest.update(chunk)
assert target.stat().st_size == expected_size, target.stat().st_size
assert digest.hexdigest() == expected, digest.hexdigest()
if target != dest:
    os.replace(str(target), str(dest))
record = {"repo": repo, "requested_revision": revision, "url": url,
          "path": str(dest), "bytes": dest.stat().st_size,
          "sha256": digest.hexdigest(), "verified": True}
Path("logs/single-demo-download.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2), flush=True)
