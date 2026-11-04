# Stage-one report: behavior cloning on one LIBERO task

Prepared November 3, 2026 for the November 4 meeting. Code author/adaptation work: Xin Wang. Scope: one LIBERO-Object task, not full paper benchmark reproduction.

## Goal and implemented work

The mentor roadmap was to understand ACT, Diffusion Policy and BAKU, run them on a small LIBERO scope, and then adapt A2A. The selected task is `pick_up_the_alphabet_soup_and_place_it_in_the_basket`.

LIBERO runs headlessly on PACE compute nodes with EGL rendering. All four author-based policies have completed training and closed-loop evaluation. Local adapters map the LIBERO demonstrations and simulator observations into each model's inputs and map model predictions back to control commands. We preserve author revisions and source patches rather than describe the algorithms as new implementations.

The demonstration file contains 50 trajectories, split by episode into 40 training and 10 validation demonstrations. Inputs are two 128×128 RGB views and nine proprioceptive values (seven arm joints and two gripper finger positions). Actions are seven controller commands: three translation components, three axis-angle rotation components and one gripper command. These are OSC controller inputs, not directly measured end-effector displacement or seven joint-angle targets. Normalization uses training demonstrations only.

ACT uses a conditional VAE to predict action chunks. DP uses the author's image-conditioned action-sequence diffusion policy. BAKU uses the author's transformer behavior-cloning agent with temporal aggregation and the single task's language embedding. A2A uses the author's flow policy starting from encoded proprioceptive history. Its original repository targets RoboVerse/Isaac workflows; our adaptation supplies LIBERO metadata, demonstrations and evaluation. The A2A history encoder accepts nine state dimensions while its future-action output stays seven-dimensional. We retain the selected author implementation's losses; paper/code differences and the controller-representation mismatch remain documented in the implementation audit.

## Evaluation and results

The common main protocol uses initial-state IDs 20–39, rollout seeds 20–39, ten all-zero warmup commands and up to 600 policy actions. Each model runs 20 complete episodes, scored by LIBERO's success check. The pinned LIBERO reference uses five zero dummy actions, so our ten-step common protocol is explicit rather than described as an exact copy of the reference.

| Method | Recipe | Seed 0 | Seed 1 |
|---|---|---:|---:|
| ACT | 10k updates, batch 16, chunk 20, execute 5 | 17/20 | 17/20 |
| DP | 3k updates, batch 16, DDPM100, execute 5 | 15/20 | 12/20 |
| DP, frozen same weights | DDPM100, execute 1 | 19/20 | 19/20 |
| BAKU | 10k updates, batch 16, query 10, temporal aggregation | 20/20 | 6/20 |
| A2A | 10k updates, batch 4, six flow steps, execute 1 | 20/20 | 0/20 |

DP's execute-one condition observes and replans after each physical action. It still uses 100 denoising iterations. The higher feedback rate improved success in both models, but its extra inference cost has not been measured. This is motivation to study action timing, not evidence for an adaptive-timing policy.

Additional frozen A2A seed-0 checks scored 19/20 with one flow step and 10/10 on previously unused initial states 40–49 of the same task. These do not establish cross-task generalization, seed stability or the paper's reported latency.

## Main adaptation issue: observation/action timing

The inspected LIBERO exporter executes command `action[t]` before saving `obs[t]`. Thus the stored observation already contains that command's effect. A policy predicting the next command should target `action[t+1]`. The adapters make this convention explicit and exclude episode-end anchors that have no next command. There are 6,156 eligible training anchors from 6,196 raw training steps.

A matched A2A seed-0 experiment held initialization, actual sample order, update budget and GPU type fixed, changing the label timing. The old-label control scored 0/20 and the next-command condition scored 20/20. This is a controlled effect in one run, not proof that timing explains every subsequent failure. The current main ACT/DP/BAKU runs also use next-command labeling; historical numbers in older notes are kept separate.

## A2A diagnosis: initial-state sensitivity

A2A's 20/20 versus 0/20 gap prompted frozen-checkpoint diagnostics instead of increasing training. Each checkpoint retained its original outcome on both V100 and RTX 6000 inference hardware. Saved training sources and normalizers matched. Selected expert-window AE/flow errors were similar, while predictions diverged on real online starting observations.

The clearest clue was the gripper: the evaluation start had a different finger opening from demonstration starts, and the failing checkpoint closed the gripper too early. Two-state input probes isolated sensitivity to gripper feedback. A physical intervention then explicitly opened the gripper during warmup.

| Same frozen A2A weights | Seed 0 | Seed 1 |
|---|---:|---:|
| Original zero warmup | 20/20 | 0/20 |
| Diagnostic open-gripper warmup | 20/20 | 20/20 |

Both diagnostic models ran on the same V100, with no training updates. This supports initial gripper state as an important trigger for these checkpoints on this task. The intervention changes evaluation initialization. It cannot replace the original scores, establish a unique training root cause or demonstrate robustness under the original protocol. The training seed and training GPU changed together, so their training-time effects remain confounded. BAKU's variation has not received an equivalent completed root-cause analysis.

## Reproducibility and limits

The repository includes exact configurations, the fixed split and dataset hash, author commits and patches, environment snapshots, historical checkpoint hashes/locations, per-rollout outcomes and action traces, source snapshots, and representative successes and failures. A standard-library verifier recomputes counts and checks the exported evidence. Heavyweight data, environments and checkpoints remain on PACE; they are not downloadable from this repository. Post-maintenance CPU validation rehashed the nine unique checkpoint files and the demonstration file, and checked all four packaged data interfaces successfully with zero updates. A PACE entry point reuses them. A completely fresh-machine setup is documented but not validated.

Interpretation is limited by one task, two training runs, only 20 evaluation episodes per main condition, repeatedly inspected development initial states, different method-specific budgets/recipes, and hardware differences between some runs. Report both training models. No broad superiority, multi-task BAKU reproduction, adaptive-policy validation, inference latency or RL result is claimed.

The next meeting should agree on acceptance of this stage-one scope, the importance of addressing BAKU/A2A stability before expanding tasks, and whether to prioritize a small timing experiment inspired by Chronos/VFRTok. The planned GitHub delivery and reading summary support that discussion; the maintenance interruption left no new research result to report.

Evidence: [run index](../results/runs.json), [results and videos](../results/README.md), [reproduction guide](REPRODUCE.md). Detailed dated source audits remain in `docs/`.
