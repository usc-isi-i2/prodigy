# SAMGPT component registry

The currently observed SAMGPT training seed is already archived in the
name-aligned `samgpt_graphcl_ladder` analysis and its source exports. This
directory registers those canonical files without duplicating them. Their
paths and SHA-256 hashes are pinned in `observed_seed.json`.

Current coverage is one complete native-GraphCL seed:

- specialist matrix: 9 sources × 9 targets = 81 cells;
- ladder: 3 orders × 9 rungs × 9 targets = 243 cells.

Two additional training seeds remain planned. They should be added as
architecture-native GraphCL evaluations with the same nine targets and fixed
unseen evaluation views. Their role is to measure seed variance and confirm
the stability of the current conclusions, not to change the pretext or force
raw metric comparability with PRODIGY.

The legacy exports do not encode the training seed identifier, so the observed
run is deliberately called `observed_seed` rather than assuming it is seed 0.
The `eval_seed` column identifies each target's fixed evaluation view and must
not be interpreted as a training seed.
