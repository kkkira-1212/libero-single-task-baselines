"""Resolve and record genuine MiniLM weights, then export fixed LIBERO demo splits."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
import numpy as np
from huggingface_hub import HfApi,snapshot_download
from sentence_transformers import SentenceTransformer
ROOT=Path(__file__).resolve().parents[1]
assert os.environ.get("SLURM_JOB_ID")
out=ROOT/"artifacts/baku-prepare"/os.environ["SLURM_JOB_ID"]
out.mkdir(parents=True,exist_ok=True)
folder=Path.home()/"scratch/robot_manipulation/baku-data"/os.environ["SLURM_JOB_ID"]
folder.mkdir(parents=True,exist_ok=True)
model_id="sentence-transformers/all-MiniLM-L6-v2"
revision=HfApi().model_info(model_id).sha
snapshot=snapshot_download(model_id,revision=revision,
    allow_patterns=["*.json","*.txt","*.safetensors","pytorch_model.bin","1_Pooling/*"])
(out/"model_provenance.json").write_text(json.dumps({"model_id":model_id,"model_revision":revision,"snapshot":str(snapshot)},indent=2)+"\n")
text="pick up the alphabet soup and place it in the basket"
encoder=SentenceTransformer(snapshot,device="cpu")
embedding=encoder.encode(text,convert_to_numpy=True).astype(np.float32)
assert embedding.shape==(384,) and np.isfinite(embedding).all() and np.linalg.norm(embedding)>0
path=folder/"task_embedding.npy"; np.save(path,embedding)
dataset=Path.home()/"scratch/robot_manipulation/datasets/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5"
digest=hashlib.sha256()
with dataset.open("rb") as f:
    for block in iter(lambda:f.read(8*1024*1024),b""): digest.update(block)
split=json.loads((ROOT/"configs/act/split.json").read_text())
assert digest.hexdigest()==split["dataset_sha256"]
subprocess.run([sys.executable,str(ROOT/"scripts/export_baku_single_task.py"),
                "--embedding",str(path),"--output-root",str(folder)],cwd=str(ROOT),check=True)
report={"completed":True,"output_root":str(folder),"model_id":model_id,"model_revision":revision,
        "task_text":text,"embedding_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
        "dataset_sha256":digest.hexdigest(),"training_episodes":split["train"],
        "validation_episodes":split["validation"],"purpose":"data_environment_embedding_export_only"}
(out/"result.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2),flush=True)
