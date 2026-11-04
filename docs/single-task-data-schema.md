# Verified single-task demonstration schema

Task: pick up the alphabet soup and place it in the basket.
Source: yifengzhu-hf/LIBERO-datasets, revision f13aa24a3da8c43c7225569f28c562979fa0e35a.
File: ~/scratch/robot_manipulation/datasets/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5
Size: 780145352 bytes.
SHA256: 42189d4415d4c51aaaf0708300653fccc39239cd3f2709079a713cd8d1678a8d (verified).

50 demonstration groups under /data, total 7808 action steps.
Episode lengths: min 136, max 196, mean 156.16. demo_0 has 148 steps.
Metadata control_freq=20 Hz, controller OSC_POSE with control_delta=true.
Nominal episode duration at that frequency is 6.8–9.8 seconds; this is metadata-derived, not measured from per-frame timestamps.
No stored train/validation mask found. No split created yet.

## Shapes for demo_0 (T=148)
| Field | Shape | dtype |
|---|---|---|
| actions | T x 7 | float64 |
| obs/agentview_rgb | T x 128 x 128 x 3 | uint8 |
| obs/eye_in_hand_rgb | T x 128 x 128 x 3 | uint8 |
| obs/ee_pos | T x 3 | float64 |
| obs/ee_ori | T x 3 | float64 |
| obs/ee_states | T x 6 | float64 |
| obs/gripper_states | T x 2 | float64 |
| obs/joint_states | T x 7 | float64 |
| robot_states | T x 9 | float64 |
| states | T x 110 | float64 |
| rewards | T | uint8 |
| dones | T | uint8 |

All 50 episodes have aligned observation/action first dimensions; all action values are finite and observed within [-1,1].
The 7 action dimensions represent a 6D end-effector pose controller input plus gripper command; they are not seven joint-angle targets.
Metadata controller input range is [-1,1], with translation output limits +/-0.05 and rotation output limits +/-0.5. Validate controller conventions before adapting a baseline.
Image metadata specifies OpenGL convention; verify orientation consistently against online observations before applying flips or augmentations.
Full simulation states are for replay/debugging; do not silently use privileged scene state as policy input.

## Next checks before training
- Inspect representative frames, gripper sign, rotation representation and reward/done boundary semantics.
- Map dataset observation names to online environment names explicitly.
- Split by complete demonstrations, not random individual frames.
- Choose baseline-specific normalization using training split only.
- Preserve episode boundaries when making action chunks; document padding.
- Stored model XML includes original author's absolute asset paths; replay must use LIBERO path rewriting/current assets.

Evidence: logs/single-demo-download.json, logs/single-demo-schema.json, logs/single-demo-summary.json.
