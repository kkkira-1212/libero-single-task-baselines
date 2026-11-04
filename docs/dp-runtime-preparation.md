# Diffusion Policy runtime preparation — 2026-10-23

Source pinned: third_party/DiffusionPolicy, commit5ba07ac6661db573af695b419a7947ecb704690f.
Use author DiffusionUnetImagePolicy/ConditionalUnet1D/MultiImageObsEncoder/LinearNormalizer; no reimplementation of denoising.

## Independent environment
scratch/robot_manipulation/dp-env, Python3.8.19, torch1.11.0+cu113/torchvision0.12, existing LIBERO requirement versions plus configs/diffusion_policy/requirements-policy.txt.
Candidate differs from upstream full Python3.9/torch1.12 conda stack; runtime checks are required. hugggingface-hub0.25.2 is selected for diffusers0.11.1 legacy cached_download compatibility, rather than mutating current ACT environment's Hub package.
Activation: source scripts/activate-dp.sh.
Installation: scripts/setup_dp_env.sh; first login-node attempt exited137/Killed during torch installation, cause not established. Moved installation/import verification to free CPU compute job14339568,16GB RAM,2CPUs,max1h. Do not say installed until logs and dp-import-check.json show success.
GPU integration job14339569 depends afterok14339568; no GPU held during dependency wait.

## Policy adapter
adapters/dp_model.py: horizon16/history2/future execution8,9D agent_pos,7D actions,two raw[0,1] RGB cameras. ImageNet normalization belongs to author encoder, not ACT preprocessing.
Train-only normalizer uses limits mode for state/action and identity for raw images. Method-specific preprocessing differs from ACT zscore and must be reported.
Separate ResNet18 copies with GroupNorm and no crop for initial interface check; no pretrained weights requested by smoke test.
DDPM100training noise timesteps,10sampling steps only for integration; EMA and formal inference/augmentation/training recipe remain to be selected.

## Combined check
scripts/check_dp_batch.py: two real training samples including boundary padding; normalizer roundtrip; epsilon loss/backward, one AdamW update; generative8-action prediction; fixed-RNG weight-reload prediction parity.
Then initialize LIBERO and execute5 predicted/clipped actions from raw live camera/state history; save screenshot. This is a functional test of an essentially untrained model, not a task-success evaluation.
Expected artifacts/dp-batch/14339569/result.json/live_step5.png and logs/dp-batch-14339569.log; diagnostic checkpoint in scratch.
Max30min,free embers,1GPU/4CPUs/24GB RAM,excluded known ECC node,email notifications. Runtime behavior unverified at submission.
Only after this gate: bundle small-fit/full-training/evaluation plan rather than repeating every ACT setup check separately.

## First runtime outcome
CPU setup14339568 succeeded (including pip check/import verification). GPU integration14339569 failed before any update because upstream nested ParameterDict usage is incompatible with torch1.11.
Scoped tensor-storage compatibility patch prepared; affine normalization algorithms untouched. CPU regression scripts/verify_dp_normalizer.py checks known values, constant dimensions, roundtrip and strict state_dict reconstruction before another GPU submission. Original ACT environment unaffected.

## Device placement lifecycle
Second integration14340553 failed before model updates: normalizer loaded after CUDA move rebuilt CPU statistics, causing CPU-vs-CUDA mismatch in roundtrip check. Initialize statistics before the full CUDA move; after checkpoint reload, explicitly restore device/dtype because dynamic normalizer reconstruction may recreate CPU tensors.
Helpers load_dp_state/assert_dp_placement enforce both paths. Retry14340945 submitted; no claim of full GPU integration success yet.
