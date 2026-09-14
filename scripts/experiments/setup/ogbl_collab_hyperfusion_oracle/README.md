# Compact HyperFusion test-oracle diagnostic

This deliberately test-informed diagnostic asks whether fixed compact AA, structure,
and joint scorers can exceed HyperFusion's reported 0.7129 test Hits@50 under the
released HyperFusion construction. It uses cosine similarities from validation and
test positive/negative score vectors to construct `H`, then `H @ H.T`, exactly as
the released script does. It is not held-out benchmark evidence.

The primary empirical-percentile interface puts heterogeneous AA scores and neural
logits on a common monotonic scale before applying the released fusion. Native scores
are a required control. Existing 2017-selected checkpoints are reused; validation
edges are appended to the graph for official test scoring. Total inference parameters
are 533,296. No retraining occurs.

Run tests and `--dry-run` locally. Run the diagnostic in a dedicated Tucker worktree
on an owned GPU, writing to a new runtime directory.
