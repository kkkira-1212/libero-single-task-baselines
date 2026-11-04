"""Versioned causal-baseline provenance and compute-node integration gate."""
import hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCES={
 "act":["adapters/libero_act.py","adapters/act_model.py","experiments/act_pilot.py","experiments/act_eval.py"],
 "dp":["adapters/libero_sequence.py","adapters/dp_model.py","experiments/dp_run.py"],
 "baku":["adapters/baku_causal.py","adapters/baku_live.py","experiments/baku_run.py","third_party/BAKU/baku/read_data/libero.py"]}
def source_hashes(method):
 paths=SOURCES[method]+["adapters/libero_causal_contract.py","adapters/baseline_causal_runtime.py","scripts/check_baseline_causal.py","scripts/activate-causal-baseline.sh"]
 return {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}
def record_run(method,out,config,training):
 from adapters.libero_causal_contract import sample_options,CONTRACT_VERSION
 options=sample_options(config)
 if config.get("sample_contract")!=CONTRACT_VERSION:return options
 job=os.environ.get("CAUSAL_GATE_JOB")
 if not job:raise RuntimeError("Causal baseline requires CAUSAL_GATE_JOB")
 path=ROOT/"artifacts/baseline-causal-check"/job/"result.json"
 gate=json.loads(path.read_text())
 assert gate["passed"] and gate["method"]==method and gate["training_updates"]==1
 current=source_hashes(method)
 assert current==gate["source_sha256"],"Sources changed after causal gate"
 out=Path(out)
 (out/"causal_protocol.json").write_text(json.dumps({"method":method,"gate_job":job,
  "sample_contract":config["sample_contract"],"options":options,"source_sha256":current},indent=2)+"\n")
 for name in current:
  target=out/("source_"+name.replace("/","__"))
  target.write_bytes((ROOT/name).read_bytes())
 return options
