"""Collect prespecified single-task runs, including missing/failed results; no best-seed filtering."""
import csv,datetime,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/"configs/main_experiments_2026-10-26.json").read_text())
ids=[str(r["job"]) for r in manifest["runs"]]
states={}
proc=subprocess.run(["sacct","-j",",".join(ids),"-n","-P","--format=JobIDRaw,State,ExitCode"],capture_output=True,text=True)
for line in proc.stdout.splitlines():
 parts=line.split("|")
 if len(parts)>=3 and parts[0] in ids:states[parts[0]]={"slurm_state":parts[1],"exit_code":parts[2]}
rows=[]
for run in manifest["runs"]:
 row=dict(run);row.update(states.get(str(run["job"]),{}))
 path=ROOT/run["report"]
 row["report_exists"]=path.exists();row["completed"]=False
 if path.exists():
  result=json.loads(path.read_text())
  row["completed"]=bool(result.get("completed",result.get("execution_ok",False)))
  for key in ("successes","completed_rollouts","success_rate","wilson95","checkpoint","checkpoint_sha256"):
   if key in result:row[key]=result[key]
 row["status"]="completed" if row["completed"] else row.get("slurm_state","missing_or_incomplete")
 rows.append(row)
stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
out=ROOT/"artifacts/main-experiments/2026-10-26";out.mkdir(parents=True,exist_ok=True)
report={"generated_at_utc":stamp,"scope":manifest["scope"],"runs":rows,
        "all_planned_comparisons_completed":all(r["completed"] for r in rows if r["role"]=="comparison")}
(out/"results.json").write_text(json.dumps(report,indent=2)+"\n")
columns=["method","protocol","seed","updates","batch_size","job","role","status","successes","completed_rollouts","success_rate","report"]
with (out/"results.csv").open("w",newline="") as f:
 writer=csv.DictWriter(f,fieldnames=columns,extrasaction="ignore");writer.writeheader();writer.writerows(rows)
lines=["# 单任务主实验进度与结果","",
       "生成时间（UTC）："+stamp,"",
       "任务：alphabet soup → basket；40 条训练示范、10 条验证示范。结果来自各自保存的评估报告。","",
       "| 方法／协议 | 种子 | 更新次数×batch | 成功／回合 | 状态 |",
       "|---|---:|---:|---:|---|"]
for r in rows:
 value=("%s/%s"%(r.get("successes","?"),r.get("completed_rollouts","?"))) if r["completed"] else "待完成"
 lines.append("| %s／%s | %s | %s×%s | %s | %s |"%(r["method"],r["protocol"],r["seed"],r["updates"],r["batch_size"],value,r["status"]))
lines+=["","解释与限制：","",
 "- 这是一个任务的适配实验，不是论文完整 suite 复现；相同初始状态被反复用于开发评估。",
 "- 不同方法的历史长度、优化器、初始化、归一化和执行方式不同，不能将差异仅归因于架构。",
 "- BAKU 两个种子差异很大，必须同时报告，不能只选择 95% 那次。",
 "- A2A 的 6 步为预定主设置，1 步为同一最终模型上的诊断对照；没有独立的论文延迟复现。",
 "- 表中包含早期失败试验和新状态确认，role 字段在 CSV/JSON 中区分；不混合计算独立测试成功率。",
 "- 缺失／失败任务保留为未完成，不能计作已完成的 0% 成功率。",
 "- 全部预定比较是否完成："+str(report["all_planned_comparisons_completed"])]
text="\n".join(lines)+"\n"
(out/"results.md").write_text(text)
(ROOT/"docs/main-experiments-2026-10-26.md").write_text(text)
print(text)
