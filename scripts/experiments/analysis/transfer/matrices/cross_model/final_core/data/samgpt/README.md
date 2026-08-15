# SAMGPT evidence registry

`three_seed/` is the canonical SAMGPT evidence for the final-core experiment.
It contains the complete native-GraphCL matrix and ladder for training seeds
39, 40, and 41 at checkpoints 20, 60, 180, and 500. The final consolidated
tables use checkpoint 500.

`three_seed/registry.json` pins the source repository and branch, evidence,
training, and evaluation commits, imported-file hashes, row counts, protocol,
seeds, checkpoints, and run-date provenance. The source export contains 3,348
physical checkpoint–target cells. At the terminal checkpoint this is 837
physical cells, which expand to 243 matrix and 729 ladder logical cells.

`observed_seed.json` is retained only as the provenance record for the earlier
single-seed export. It is superseded by `three_seed/` and is not read by the
canonical table builder or verifier.

Training seeds and evaluation-view seeds are distinct. `training_seed` is
39/40/41; `eval_seed` is target-specific under the fixed evaluation-view rule
and must not be interpreted as a training seed.
