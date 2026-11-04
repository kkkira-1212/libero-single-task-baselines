# Single-task demonstration inspection plan

Source inspected: official LIBERO checkout commit 8f1084e3132a39270c3a13ebe37270a43ece2a01.
- Download implementation: libero/libero/utils/download_utils.py.
- Loader implementation: libero/lifelong/datasets.py.
- Official loader uses robomimic SequenceDataset, observations plus actions, without loading next_obs. Sequence padding is enabled; respect episode boundaries when designing action chunks.
- Hugging Face repository selected by official code: yifengzhu-hf/LIBERO-datasets.
- Target: libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5.
- Hub metadata size: 780145352 bytes (about 780 MB / 744 MiB).
- Expected SHA256 from Hub LFS metadata: 42189d4415d4c51aaaf0708300653fccc39239cd3f2709079a713cd8d1678a8d.
- Downloading the standard suite command would fetch all tasks in the suite; prefer one-file download for this sprint.
- No demonstration file downloaded yet.

After download authorization:
1. Download only the selected HDF5 to the scratch datasets directory and record repository revision/checksum.
2. Run scripts/inspect_demo.py to inspect actual episode counts, observation shapes/dtypes, action ranges, attributes and masks. Script has syntax validation only; actual schema remains unverified.
3. Inspect representative RGB frames and timestamps/controller metadata. Do not assume recorded data timing equals an arbitrary environment setting.
4. Split whole demonstrations into training/validation sets with a fixed seed; avoid overlapping episode windows across splits.
5. Establish observation/action meanings and normalization from actual metadata and controller code; fit any learned statistics on training data only.
6. Record the protocol before training any baseline.
