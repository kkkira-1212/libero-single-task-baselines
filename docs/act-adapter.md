# ACT/LIBERO interface adapter

## Provenance
Author model: third_party/ACT, original commit 742c753c0d4a5d87076c8f69e5628c79a8cc5488.
Local changes saved in patches/act-libero-interface.patch. Retains author CVAE/Transformer/action-chunk implementation and loss convention.
Upstream patch separates state_dim and action_dim, preserving original 14D defaults; adds explicit pretrained-backbone switch; removes unused IPython debug imports from four model modules. No package installation required.
Fresh source restoration: checkout the recorded commit, then apply the saved patch with git apply in third_party/ACT (do not apply twice).

## Data contract
adapters/libero_act.py reads existing HDF5 without converting/copying dataset:
- State: seven joint positions followed by two gripper positions (9D).
- Images: agentview then wrist, 128x128 RGB; HWC uint8 -> CHW float32.
- Applies ImageNet mean/std once. This feeds the model directly, NOT the author's ACTPolicy wrapper (which would normalize a second time).
- Uses stored image orientation, with the identical transform on raw online observations. Display-oriented PNG flipping is separate.
- Actions: seven original LIBERO controller inputs, not joint-position targets. Decode with saved training means/std; clip to environment [-1,1].
- Returns future actions from current t, stopping at episode end; normalized-space zero padding and bool is_pad. No real-robot one-frame offset.
- Files opened per access; no shared live HDF5 handle across loader workers.
- Every real training timestep is one sample; this is explicit uniform-over-timesteps sampling, differing from the author's episode-uniform random-timestep loader.
- Chunk size 20 is an initial interface-test setting, not a claim of optimal or paper-matched hyperparameters.

## Reproducible protocol files
configs/act/interface.json describes preprocessing.
configs/act/split.json stores seed-0 whole-episode split (40 train / 10 validation) and dataset checksum.
configs/act/normalization.json stores state/action statistics fit only on training demonstrations with std floor 0.01.
Validation losses may inform checkpoint selection; online evaluation states/horizon and success-rate target are not frozen yet.
No original dataset modified.

## Model construction and loss
adapters/act_model.py calls the author model builder without the author's global CLI parsing or implicit CUDA.
Defaults retain author-style network sizes; model structure check uses randomly initialized ResNet18 to avoid weight downloads.
Actual training must explicitly decide pretrained weights before the run and save that decision.
act_loss matches author's padded-mean masked L1 + KL convention, including the original padded denominator. Do not silently substitute a different mask reduction.
No training runner, temporal aggregation or full rollout adapter has been validated yet.

## Checks
Run from project root after source scripts/activate-libero.sh:
- OPENBLAS_NUM_THREADS=1 python scripts/verify_act_adapter.py
- OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/verify_act_model_structure.py
The former checks real episode boundaries, pad masks, split isolation, train-only stats and action roundtrip. Offline/online preprocessing parity uses identical offline inputs mapped to online key names; actual live-environment parity remains a separate test.
The latter constructs a CPU model to inspect dimensions only; no forward/backward, GPU or training.
Next compute milestone: real batch forward/backward, padded-loss behavior, and checkpoint save/reload on free embers, followed by a small-data fitting diagnostic. No success-rate claim until closed-loop evaluation.

## GPU integration result
Job 14272806 on Quadro RTX6000 passed real-batch forward/backward, one optimizer step, padded-loss invariance and strict model-weight reload prediction parity. Report: artifacts/act-batch/14272806/result.json.
No pretrained weights, fitting study, optimizer resume or online rollout was tested. Earlier statements about unvalidated model operations are superseded for this specific allocation and batch only.
