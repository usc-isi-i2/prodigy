# Standard Cora feature-only link prediction

This campaign starts from the classic Planetoid-style Cora artifact: 2,708
papers, 5,278 unique undirected citation pairs, and 1,433-dimensional binary
bag-of-words node features. It does not use GTE features, message passing,
pretraining, node labels, or the repository's earlier LP results.

## Frozen protocol

- Seed: 0.
- Split the unique undirected citation pairs 85/5/10 into train/validation/test.
  Split pairs before expanding orientations, so a reverse edge cannot cross splits.
- Sample one unique random unordered nonedge per positive. Validation and test
  negatives are fixed; training negatives are fixed as well to make all learned
  arms see the exact same examples.
- Evaluate three rungs on the identical validation/test pairs:
  1. `raw_cosine`: cosine of the original 1,433-D binary features; no training.
  2. `linear_cosine`: shared learned `Linear(1433, 128)` node map, L2 normalize,
     cosine score.
  3. `nonlinear_mlp_cosine`: shared `Linear(1433,256) -> ReLU -> Linear(256,128)`
     node map, L2 normalize, cosine score.
- Learned arms use a scalar learned logit scale and bias, full-batch Adam,
  learning rate 0.01, weight decay 0.0005, at most 500 epochs, and patience 50.
- Each learned arm logs every raw epoch value to a separate W&B offline run in
  the output directory. No smoothing or online synchronization is used.
- Select each learned arm's checkpoint by validation ROC-AUC (earliest epoch wins
  a tie). Do not compute test metrics until both selections are frozen.
- Primary metric: test ROC-AUC. Secondary: average precision. Also report the
  raw cosine rung and exact split/pair fingerprints.
- This is a one-seed matched pilot. It is not a replicated estimate.

Random negatives are intentional: this is the conventional minimal baseline,
not the repository's stricter degree-matched transfer evaluator.

## Run on Tucker

```bash
export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
python scripts/experiments/setup/cora_standard_mlp_lp/run.py \
  --graph /dataMeR1/phil/data/cora/raw/graph_com_tag/processed_data.pt \
  --out /dataMeR1/phil/gfm/cora_standard_mlp_lp/seed0
```

The output directory must not already exist. `protocol.json`, `pairs.npz`,
`selection.json`, `results.json`, and the two selected checkpoints form the run
record. Offline W&B run directories are retained with the Tucker runtime output.
Render the exact unsmoothed curves with `plot_curves.py`. The analysis copy belongs in
`scripts/experiments/analysis/baselines/cora_standard_mlp_lp/data/`.
