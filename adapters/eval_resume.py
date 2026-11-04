"""Strict continuation of independent, fully completed evaluation trials."""
import json,shutil
from pathlib import Path

def restore_trials(parent_path,out,config,checksum,sampling_steps,execution_horizon):
    parent_path=Path(parent_path);parent=json.loads(parent_path.read_text())
    assert parent["config"]==config,"Resume training config mismatch"
    assert parent["checkpoint_sha256"]==checksum,"Resume checkpoint mismatch"
    assert parent.get("sampling_steps",parent["config"].get("inference_steps"))==sampling_steps,"Resume sampling step mismatch"
    assert parent["execution_horizon"]==execution_horizon,"Resume action execution mismatch"
    assert parent["weights"]=="EMA"
    expected=dict(zip(config["rollout_initial_state_ids"],config["rollout_seeds"]))
    rows=[];seen=set()
    for original in parent["rollouts"]:
        if not original.get("completed",False):continue
        row=dict(original);i=row["initial_state_id"]
        assert i in expected and row["seed"]==expected[i] and i not in seen
        assert 0<row["steps"]<=config["rollout_horizon"]
        assert isinstance(row["success"],bool)
        for name in ("trial_%s.gif"%i,"trial_%s_actions.json"%i):
            source=parent_path.parent/name
            assert source.is_file() and source.stat().st_size>0,"Missing completed-trial evidence"
            shutil.copy2(source,Path(out)/name)
        row.setdefault("source_job_id",parent["job_id"])
        rows.append(row);seen.add(i)
    return rows
