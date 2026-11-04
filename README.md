# LIBERO single-task behavior cloning study

Author-code-based adaptations of **ACT, Diffusion Policy, BAKU, and A2A** to one LIBERO-Object task: **pick up the alphabet soup and place it in the basket**.

This repository is a stage-one research deliverable. It contains adaptation code, pinned upstream revisions and patches, configurations, recorded evaluation evidence, representative animations, and a short report. The experiments cover one task and two training runs per method. They do not reproduce the full published benchmarks or establish an equal-compute ranking.

## Results

All main evaluations use the corrected next-command labeling, the same 40/10 demonstration split, 20 initial states (IDs 20–39), ten zero-command warmup steps, and a 600-step cap. Each column is a separately trained model. Method-specific recipes differ.

| Method / execution | Updates | Seed 0 | Seed 1 |
|---|---:|---:|---:|
| ACT, chunk 20 / execute 5 | 10,000 | 17/20 (85%) | 17/20 (85%) |
| Diffusion Policy, DDPM100 / execute 5 | 3,000 | 15/20 (75%) | 12/20 (60%) |
| DP, same weights / execute 1 | same 3k weights | 19/20 (95%) | 19/20 (95%) |
| BAKU, temporal aggregation | 10,000 | 20/20 (100%) | 6/20 (30%) |
| A2A, 6 flow steps / execute 1 | 10,000 | 20/20 (100%) | 0/20 (0%) |

**DP execute 1 changes feedback frequency, not its 100 denoising steps.** A2A's one-flow-step check is a separate experiment: the seed-0 checkpoint scored 19/20. That same checkpoint scored 10/10 on additional initial states from the same task. No inference-latency reproduction is claimed.

### A2A initialization diagnosis

| Frozen model condition | Seed 0 | Seed 1 |
|---|---:|---:|
| Original zero-command warmup | 20/20 | 0/20 |
| Diagnostic explicitly open gripper during warmup | 20/20 | 20/20 |

The diagnostic uses unchanged weights, the same 20 initial-state IDs, and the same allocated V100 for both models. It deliberately changes initialization and **does not replace the original results**. Initial gripper-state sensitivity is an important failure trigger for these checkpoints. Training GPU and training seed remain confounded. BAKU's cross-run variation remains unresolved.

## Start here

- [Short report](docs/STAGE_ONE_REPORT.md): scope, adaptation work, results, issues and limitations.
- [Reproduction guide](docs/REPRODUCE.md): data, environments, pinned author sources, training and evaluation.
- [Machine-readable results](results/runs.json) and [CSV](results/summary.csv): every reported run, setting, checkpoint checksum, and evidence location.
- [Results and videos](results/README.md): representative successes and failures, plus protocol distinctions.
- [Delivery status](docs/DELIVERY_STATUS.md): what has actually been verified and what remains pending.

## Inspect results without a GPU

Python 3.8+ and the standard library are sufficient to verify the exported evaluation records:

```bash
python3 tools/verify_delivery.py
```

This recomputes success counts from per-rollout records, validates all 290 action traces, verifies copied source snapshots and file-integrity records, and checks the result table. It does not execute a policy or rehash unpublished checkpoint files. See `results/verification.json` for the latest recorded check.

## Run on the existing PACE environment

The tested environment is Linux/PACE with existing scratch virtual environments, demonstrations, and checkpoints. New-machine installation has not been tested. Author repositories and large data/weights are not bundled.

```bash
python3 tools/bootstrap_upstream.py
python3 tools/prepare_runtime.py
mkdir -p logs
sbatch tools/pace_job.sbatch eval dp 0 --execute-one
```

This performs the full 20-rollout DP seed-0 frozen-weight evaluation, using 100 denoising steps and executing one physical action before replanning. It needs the checkpoint indexed in `results/runs.json`. Read [the reproduction guide](docs/REPRODUCE.md) before submitting compute. It explains relocating paths and using the preserved source gates.

## Repository map

| Path | Purpose |
|---|---|
| `adapters/` | LIBERO data, observation/action interfaces, model construction, causal timing |
| `experiments/` | Actual policy training and closed-loop evaluation |
| `tools/` | Curated repository bootstrap, PACE entry point, and evidence verifier |
| `configs/baselines_causal/` | Current ACT/DP/BAKU configurations and job manifest |
| `configs/a2a/` | A2A causal configurations, frozen manifests and diagnostics |
| `results/` | Main and additional results, per-rollout evidence, animations, archived gates |
| `upstream/` | Author URLs/commits, exact patches and original license texts |
| `environments/` | Recorded environment snapshots and demonstration provenance |
| `docs/` | Current delivery documents plus dated research notes |
| `scripts/` | Original research scripts; many `.sbatch` files preserve original absolute PACE paths |

Use `tools/pace_job.sbatch` as the curated entry point. Dated notes and historical configs intentionally retain failed/pending observations from their creation dates. The tables above, `results/runs.json`, and the stage-one report are the current summary.

## Source attribution

| Component | Author source | Pinned commit |
|---|---|---|
| LIBERO | [Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) | `8f1084e3132a39270c3a13ebe37270a43ece2a01` |
| ACT | [tonyzhaozh/act](https://github.com/tonyzhaozh/act) | `742c753c0d4a5d87076c8f69e5628c79a8cc5488` |
| Diffusion Policy | [real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy) | `5ba07ac6661db573af695b419a7947ecb704690f` |
| BAKU | [siddhanthaldar/BAKU](https://github.com/siddhanthaldar/BAKU) | `47e367059191de30ac4b6353ff5615fdef11d8c5` |
| A2A | [JIAjindou/A2A_Flow_Matching](https://github.com/JIAjindou/A2A_Flow_Matching) | `a5792ecf4e7f8fa4d85fe66ea9a50618138f925c` |

The mentor's ACT link referenced ALOHA; these experiments use the author's ACT repository shown above. Original author license texts are retained in `upstream/licenses/`. Third-party snapshots and patches retain their upstream attribution. This repository does not claim authorship of those algorithms or relicense upstream material.
