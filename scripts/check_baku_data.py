"""Audit exported timing/flip/split and exercise the unchanged author BCDataset."""
import argparse,hashlib,json,os,pickle,random,sys
from pathlib import Path
import h5py,numpy as np,torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"third_party/BAKU/baku"))
from read_data.libero import BCDataset
parser=argparse.ArgumentParser(); parser.add_argument("--prepare-job",required=True); args=parser.parse_args()
assert os.environ.get("SLURM_JOB_ID")
out=ROOT/"artifacts/baku-data-check"/os.environ["SLURM_JOB_ID"];out.mkdir(parents=True,exist_ok=True)
prep=json.loads((ROOT/"artifacts/baku-prepare"/args.prepare_job/"result.json").read_text())
assert prep["completed"]
split=json.loads((ROOT/"configs/act/split.json").read_text())
assert not set(split["train"]) & set(split["validation"])
task="pick_up_the_alphabet_soup_and_place_it_in_the_basket"
datafile=Path.home()/"scratch/robot_manipulation/datasets/libero_object"/(task+"_demo.hdf5")
report={"passed":False,"purpose":"BAKU_author_data_interface_only","prepare_job":args.prepare_job,
        "model_forward_backward":False,"online_rollouts":0,"subsets":{}}
with h5py.File(datafile,"r") as source:
    for subset in ("train","validation"):
        file=Path(prep["output_root"])/subset/"libero_object"/(task+".pkl")
        with file.open("rb") as f: exported=pickle.load(f)
        assert exported["episode_ids"]==split[subset]
        assert len(exported["actions"])==len(split[subset])
        for index,name in enumerate(split[subset]):
            original=source["data"][name]; obs=exported["observations"][index]
            np.testing.assert_array_equal(exported["actions"][index],original["actions"][:].astype(np.float32))
            for key,src in (("pixels","agentview_rgb"),("pixels_egocentric","eye_in_hand_rgb")):
                for t in (0,len(original["actions"])-1):
                    np.testing.assert_array_equal(obs[key][t],original["obs"][src][t][::-1])
            for key in ("joint_states","gripper_states"):
                np.testing.assert_array_equal(obs[key],original["obs"][key][:].astype(np.float32))
        del exported
        random.seed(0);np.random.seed(0);torch.manual_seed(0)
        ds=BCDataset(path=str(Path(prep["output_root"])/subset),suite="libero_object",
                     scenes=["object"],tasks=[{"object":[task]}],num_demos_per_task=len(split[subset]),
                     obs_type="pixels",history=True,history_len=2,prompt="text",
                     temporal_agg=True,num_queries=20,img_size=128)
        batch=next(iter(torch.utils.data.DataLoader(ds,batch_size=2,num_workers=0)))
        shapes={k:list(v.shape) for k,v in batch.items()}
        assert shapes=={"pixels":[2,2,3,128,128],"pixels_egocentric":[2,2,3,128,128],
                        "proprioceptive":[2,2,9],"actions":[2,2,20,7],"task_emb":[2,384]},shapes
        assert all(torch.isfinite(v).all() for v in batch.values())
        assert batch["pixels"].min()>=0 and batch["pixels"].max()<=1
        report["subsets"][subset]={"episodes":len(split[subset]),"shapes":shapes,"pkl_sha256":hashlib.sha256(file.read_bytes()).hexdigest()}
        del ds,batch
report["passed"]=True
(out/"result.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2),flush=True)
