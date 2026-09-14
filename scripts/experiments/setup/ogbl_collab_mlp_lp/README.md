# Official ogbl-collab feature-only LP ladder

This campaign replaces the ad hoc Cora LP split with OGB's fixed temporal
`ogbl-collab` protocol. It asks how much future collaboration prediction is
available in the supplied 128-dimensional author features without graph message
passing, node IDs, edge weights, or validation-edge reuse.

The machine-readable contract is [`protocol.yaml`](protocol.yaml). Do not change
its scientific settings after inspecting production test results; a changed
setting is a new tagged campaign.

Operational departures from that contract are recorded append-only in
[`DEVIATIONS.md`](DEVIATIONS.md).

## Frozen comparison

The three matched arms are:

1. `raw_cosine`: cosine of the supplied 128-D author features.
2. `linear_cosine`: shared `Linear(128, 128)` map, L2 normalization, cosine.
3. `nonlinear_mlp_cosine`: shared `Linear(128, 256) -> ReLU -> Linear(256, 128)`
   map, L2 normalization, cosine.

The learned arms include a learned positive score scale and bias. They receive
the same shuffled positives and random-negative stream within every seed and
epoch. One epoch is one pass over all 1,179,052 official training positives,
which is 18 optimizer updates at batch size 65,536. Validation runs once per
epoch—not after every update. Checkpoints are selected by official validation
Hits@50, with earliest epoch breaking ties. Test scoring begins only after both
learned selections for a seed have been written to `selection_frozen.json`.

Production uses seeds 0-2. Official test Hits@50 mean and sample standard
deviation are the primary result. Hits@10, Hits@100, ROC-AUC, AP, and score
summaries are secondary.

## Intended recurrence and the predeclared diagnostic

`ogbl-collab` is a temporal multigraph and predicts 2019 collaboration events
from older history. Training itself can contain repeated pair events from
different years, and some test
pairs may have collaborated before, which is intended by the benchmark but is
not the same estimand as predicting a brand-new relationship. Before inspecting
scores, this campaign therefore predeclares official-negative Hits@K broken out
for test positives that are repeated versus novel relative to training, and
relative to all pre-2019 history. The overall official Hits@50 remains the
headline number.

Validation edges are not used as labels or model input. This deliberately avoids
the optional OGB Collab exception that permits validation-edge reuse after tuning.

## Tucker workflow

The official dataset lives at `/dataMeR1/phil/data/ogb`. Validate its exact
counts, years, feature shape, split fingerprint, official-negative self-pair
counts, and planned update count before launching. The evaluator consumes the
fixed official negative arrays exactly as supplied; the campaign records but
does not silently clean them. Because labels are year-specific events, a negative
for one year may legitimately be a positive in another year. The audit rejects
same-year positive/negative collisions and records cross-year overlaps rather
than incorrectly treating them as leakage:

```bash
python scripts/experiments/setup/ogbl_collab_mlp_lp/run.py \
  --dataset-root /dataMeR1/phil/data/ogb \
  --out /tmp/not-created-in-dry-run \
  --seed 0 --device cuda:0 --dry-run
```

Smoke validation is separate and can never be admitted as production evidence:

```bash
python scripts/experiments/setup/ogbl_collab_mlp_lp/run.py \
  --dataset-root /dataMeR1/phil/data/ogb \
  --out /dataMeR1/phil/gfm/ogbl_collab_mlp_lp/smoke_seed0 \
  --seed 0 --device cuda:0 --epochs 2 --patience 2 \
  --wandb-mode disabled --run-tag smoke
```

After the smoke passes, use three dedicated tmux jobs on GPUs 0-2:

```bash
bash scripts/experiments/setup/ogbl_collab_mlp_lp/run_tucker.sh 0 0
bash scripts/experiments/setup/ogbl_collab_mlp_lp/run_tucker.sh 1 1
bash scripts/experiments/setup/ogbl_collab_mlp_lp/run_tucker.sh 2 2
```

Each seed writes `protocol.json`, two checkpoints, `selection_frozen.json`,
`results.json`, and local W&B records under
`/dataMeR1/phil/gfm/ogbl_collab_mlp_lp/official_v1/seed<seed>/`.

Once every seed is complete, validate and aggregate:

```bash
python scripts/experiments/setup/ogbl_collab_mlp_lp/aggregate.py \
  --root /dataMeR1/phil/gfm/ogbl_collab_mlp_lp/official_v1 \
  --out /dataMeR1/phil/gfm/ogbl_collab_mlp_lp/official_v1/aggregate.json
```

Admitted evidence and findings belong under
`scripts/experiments/analysis/baselines/ogbl_collab_mlp_lp/` only after the
completion and provenance checks pass.

## Post-hoc test-oracle diagnostic

The separately tagged `test_oracle_diagnostic_v1` campaign deliberately evaluates
the official test panel at every validation epoch. It retrains the frozen learned
arms with unchanged optimization and validation-only early stopping, then measures
the gap between test Hits@50 at the validation-selected epoch and the maximum test
Hits@50 that could be chosen retrospectively. These runs are non-admissible as
benchmark evidence and must never replace `official_v1` results:

```bash
bash scripts/experiments/setup/ogbl_collab_mlp_lp/run_test_oracle_tucker.sh 0 0
bash scripts/experiments/setup/ogbl_collab_mlp_lp/run_test_oracle_tucker.sh 1 1
bash scripts/experiments/setup/ogbl_collab_mlp_lp/run_test_oracle_tucker.sh 2 2
```
