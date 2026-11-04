# Delivery status — November 4, 2026

- Completed historical training/evaluation: ACT, DP, BAKU and A2A on one task. Export includes15conditions/290episodes, complete actions, source snapshots and representative animations.
- Independent exported-evidence verification: passed locally. See `results/verification.json`.
- Author code: exact revisions and generated patches recorded, original licenses retained.
- Repository tools: curated PACE entry point added; Python compilation and shell syntax checked. Post-maintenance CPU check15038964 COMPLETED0:0 in11m09s. ACT/DP/BAKU/A2A real-data interface checks all passed with zero updates. GPU15039257 failed before model loading: the new smoke script omitted CAUSAL_GATE_JOB. The script now selects the recorded DP gate from the baseline manifest and prepares runtime paths. A direct source-guard check passed (gate14497123,7files). Retry15058569 COMPLETED0:0 in7m40s onQuadroRTX6000:checkpointSHA matched, two policy calls,10finite simulator actions and128×128RGB EGL rendering passed,zero updates. Evidence:results/maintenance/dp_smoke_15058569/. No benchmark score is changed.
- Checkpoint files: retained on PACE, paths/historical checksums indexed. All9unique retained checkpoints and the demonstration file were rehashed successfully on the compute node. See results/maintenance/checks_2026-11-03.json.
- GitHub publication: https://github.com/kkkira-1212/libero-single-task-baselines is public, after explicit user authorization. A fresh clone passed the exported-evidence audit.
- No new training or adaptive-timing research experiment was run for this deliverable. Maintenance prevented the earlier planned work; existing findings are presented with their actual limits.
- Unresolved: BAKU cross-run variation, A2A original-initialization robustness, training-GPU/seed confounding, full clean-machine installation, downloadable checkpoint distribution, measured latency, suite/task expansion.
- User's next step: systematic reading summary and a brief oral walkthrough before sharing the repository with mentors. No mentor message has been sent by the assistant.
