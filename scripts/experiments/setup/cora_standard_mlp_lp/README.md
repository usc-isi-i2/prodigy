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

## Revised resampled-negative protocol

The follow-up requested after inspecting the raw pilot curves changes exactly two
training settings while preserving the seed-0 positive split and fixed validation/test
panels:

- learning rate `0.001` instead of `0.01`;
- a fresh deterministic set of 4,486 unique training nonedges at every epoch,
  using negative-stream seed 1000, with the same epoch-indexed stream for both arms.

It allows 1,000 epochs with patience 100. Each selection-history row stores the
epoch's negative-set fingerprint so matched prefixes can be verified directly.

```bash
python scripts/experiments/setup/cora_standard_mlp_lp/run.py \
  --graph /dataMeR1/phil/data/cora/raw/graph_com_tag/processed_data.pt \
  --out /dataMeR1/phil/gfm/cora_standard_mlp_lp/seed0_lr1e3_resampled \
  --learning-rate 0.001 --epochs 1000 --patience 100 \
  --resample-train-negatives --train-negative-seed 1000 \
  --run-tag lr1e3_resampled
```

This is a revised protocol, not a replacement for the fixed-negative pilot.

## Validation-cadence protocol (v3)

The next iteration keeps the lower learning rate, epoch-wise resampled negatives,
and fixed seed-0 pair panels from the preceding follow-up. It changes validation
from every optimizer update to every 5 updates and uses patience 20 validation
checks (100 updates without improvement). Maximum training remains 1,000 updates.

Validation and test report ROC-AUC, average precision, binary cross-entropy, Brier
score, accuracy, balanced accuracy, precision, recall, and F1. Thresholded metrics
use a fixed probability cutoff of 0.5. Because the pair panels are balanced by
sampling, calibration and thresholded metrics describe that sampled distribution,
not Cora's natural edge prevalence.

This run uses W&B online because the user explicitly requested immediate monitoring:

```bash
python scripts/experiments/setup/cora_standard_mlp_lp/run.py \
  --graph /dataMeR1/phil/data/cora/raw/graph_com_tag/processed_data.pt \
  --out /dataMeR1/phil/gfm/cora_standard_mlp_lp/seed0_v3_val5 \
  --learning-rate 0.001 --epochs 1000 --patience 20 --val-interval 5 \
  --resample-train-negatives --train-negative-seed 1000 \
  --wandb-mode online --run-tag v3_val5
```
