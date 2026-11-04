# Reproduction guide

## Tested scope

Recorded experiments ran on PACE, with Python3.8.19 / torch1.11.0+cu113 for ACT, DP and BAKU, and Python3.10.10 / torch2.2.2+cu118 for A2A. LIBERO uses robosuite1.4 / MuJoCo2.3.7 and EGL. Exact snapshots are in `environments/`. These are recorded installed environments, not a claim that a clean installation has been retested on every machine.

The curated wrapper keeps original model/data/loss code byte-identical so that archived source gates remain meaningful. Original `.sbatch` scripts in `scripts/` preserve historical absolute paths and email/account settings. Prefer `tools/pace_job.sbatch` and override resource/account settings appropriately for another allocation. Existing experiments used only free `gts-agarg35` / `embers` resources.

## 1. Inspect exported results first

```bash
python3 tools/verify_delivery.py
```

No data, author repositories, torch or GPU is needed. This checks 15 exported conditions / 290 episodes, their actions, source snapshots and summary. The A2A old-label control is intentionally a different sample contract. It is not part of the main cross-method table.

## 2. Fetch author sources

```bash
python3 tools/bootstrap_upstream.py
python3 tools/prepare_runtime.py
```

The bootstrap clones the five repositories in `upstream/manifest.json`, checks out their recorded commits and applies the recorded patches. It refuses unexpected tracked modifications rather than resetting a checkout. It does not install packages. `prepare_runtime.py` generates local LIBERO paths and copies archived gates into the ignored runtime `artifacts/` tree. In the existing PACE delivery directory, author repositories may be symlinked to the already verified checkouts; use `--check-only` to verify them.

ACT patches make dimensions explicit and backbone pretraining selectable; DP patches support dynamic normalizer state restoration in the selected old torch environment; A2A patches separate history-state dimension9 from action dimension7. BAKU and LIBERO have no tracked source patches in this delivery. All other adaptation code is local to this repository.

## 3. Environments and demonstration file

The existing PACE virtual environments are:

| Method | Python environment |
|---|---|
| ACT | `~/scratch/robot_manipulation/libero-env` |
| DP | `~/scratch/robot_manipulation/dp-env` |
| BAKU | `~/scratch/robot_manipulation/baku-env` |
| A2A | `~/scratch/robot_manipulation/a2a-env` |

For a new machine, provision the indicated Python/torch combination, use the snapshots under `environments/` as version references, and review the original `scripts/setup_*` scripts and `configs/*/requirements*` for package installation. Editable/local file entries in snapshots must be relocated. Install the pinned local LIBERO checkout with `pip install --no-deps -e third_party/LIBERO`. Install/run dependency checks and data gates before training. The repository does not bundle virtual environments or silently install Isaac Sim. A fully clean-machine recipe remains unverified.

Expected demonstration location:

```text
~/scratch/robot_manipulation/datasets/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5
```

Source: `yifengzhu-hf/LIBERO-datasets`, revision `f13aa24a3da8c43c7225569f28c562979fa0e35a`.

[Download the recorded demonstration file](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/resolve/f13aa24a3da8c43c7225569f28c562979fa0e35a/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5).

Expected bytes:780145352. SHA256:`42189d4415d4c51aaaf0708300653fccc39239cd3f2709079a713cd8d1678a8d`.

`configs/act/split.json` holds the fixed40/10 episode split. `configs/act/normalization.json` holds ACT training-only statistics. Current next-command configs use `libero-poststep-next-command-v1`. Do not switch to historical same-index configs while comparing the current table.

BAKU additionally uses PKL exports with one vertical image flip and a fixed384D MiniLM language embedding. The embedding and its author-model revision are in `results/data_provenance/baku/`. To regenerate exports in a compute allocation with the BAKU environment:

```bash
python scripts/export_baku_single_task.py \
  --embedding results/data_provenance/baku/task_embedding.npy \
  --output-root /YOUR/SCRATCH/baku-data
```

`configs/baselines_causal/baku_seed*.json` preserve the original `data_root`. On another account, copy these configs and change only the data location, check the PKL checksums, then train new models with those configs. Evaluation validates the config saved inside the checkpoint, so relocating data/checkpoints needs deliberate config/path handling; do not quietly bypass those checks. The existing PACE account already has the recorded exports.

## 4. Data/integration checks

From the repository root, create `logs/` before `sbatch`:

```bash
mkdir -p logs
sbatch tools/pace_job.sbatch data-check dp 0
sbatch tools/pace_job.sbatch gate dp 0
```

Replace `dp` with `act`, `baku` or `a2a`. Data-check inspects real demonstration windows and labels without optimizer updates. A GPU gate checks model operations and reload, and performs a disposable update; passing a gate is not a measured success rate. The `gate` job ID can be supplied to later runs using `--gate-job JOB_ID`. The source-hash guard must pass. If model, data or guarded source code changes, rerun the appropriate gate.

## 5. Train a new model

```bash
sbatch tools/pace_job.sbatch train act 0
sbatch tools/pace_job.sbatch train dp 0
sbatch tools/pace_job.sbatch train baku 0
sbatch tools/pace_job.sbatch train a2a 0
```

Use `1` for the second training seed. These are substantial new training jobs, not needed merely to inspect the stage-one deliverable. Baselines use the archived source gates by default. A2A uses the preserved integration report and paired causal gate. After a new integration check or any guarded source change, pass its new gate ID and update the configuration's integration provenance as necessary.

Training outputs go to method-specific scratch directories keyed by the new Slurm job ID. The main historical checkpoints are final predetermined updates, not selected by online success:ACT10k,DP3kEMA,BAKU10k,A2A10kEMA. Do not interpret an incomplete job's checkpoint as a completed recipe.

## 6. Evaluate the recorded frozen checkpoints

These commands need the checkpoint files indexed in `results/runs.json`; the repository publishes hashes and paths, not the weight files. On the existing PACE account, the files are retained under scratch.

```bash
sbatch tools/pace_job.sbatch eval act 0
sbatch tools/pace_job.sbatch eval dp 0
sbatch tools/pace_job.sbatch eval dp 0 --execute-one
sbatch tools/pace_job.sbatch eval baku 0
sbatch tools/pace_job.sbatch eval a2a 0
```

Each command runs20episodes. Use seed1 to evaluate the other model. ACT/DP/BAKU accept `--checkpoint /path/to/new/latest.pt` for a compatible checkpoint and `--config` where needed; a checkpoint trained with a different config must be paired with that original config. A2A uses `configs/a2a/frozen_causal_seed{0,1}_10000.json`; to relocate a checkpoint, deliberately update a working copy's checkpoint path while preserving the hash and original training config.

A2A diagnostic open-gripper evaluation must be requested explicitly:

```bash
python3 - <<'PY'
import json
from pathlib import Path
p=Path('artifacts/open-gripper-protocol.json')
p.parent.mkdir(exist_ok=True)
p.write_text(json.dumps({'purpose':'diagnostic_changed_initialization','sampling_steps':6,
                         'execution_horizon':1,'settle_gripper_command':-1.0}))
PY
sbatch tools/pace_job.sbatch eval a2a 1 --protocol artifacts/open-gripper-protocol.json
```

Keep this score separate from the original zero-warmup score. The frozen evaluator inserts and checks the checkpoint hash from its manifest. To inspect the one-flow-step condition, use `--protocol configs/a2a/confirm_seed0_steps1.json` with A2A seed0.

## 7. Maintenance/repository verification

```bash
sbatch tools/verify_pace.sbatch
```

This read-only CPU job verifies the archived upstream revisions/patches, exported results, retained checkpoint/dataset bytes, and real-data interfaces for all four methods. It performs no optimizer updates or scored rollouts. The current status and job ID are in `docs/DELIVERY_STATUS.md`. The short frozen-DP GPU/EGL check is tools/smoke_dp.sbatch. It runs two policy calls and at most10physical actions, with no scored task-success claim. Its actual status is recorded separately there.
