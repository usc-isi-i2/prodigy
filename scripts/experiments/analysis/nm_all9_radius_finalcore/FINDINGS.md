# All-nine radius-controlled neighbor matching

## Result

Constraining training episodes by center-node radius did not improve accuracy on
matched close-radius tasks. After 2,500 optimizer updates, `radius_mix` nearly matched
the global control on radius-2 and radius-3 tests, but gave up 3.45 percentage points
on global-center episodes. The all-close `close_only` arm was worse on every panel.

All reported scores are 30-way query classification accuracy, not ROC-AUC. Chance is
1/30 = 3.33%.

| frozen test panel | `global` | `radius_mix` | `close_only` |
|---|---:|---:|---:|
| radius 2 | **17.92%** | 17.68% | 17.24% |
| radius 3 | **18.85%** | 18.79% | 18.25% |
| global | **62.25%** | 58.80% | 50.40% |
| within source | **26.28%** | 25.53% | 24.57% |

Values are means over training seeds 0, 1, and 2. Every one of the nine validation
selections chose checkpoint 2,500.

## Setup and comparison contract

The three arms differ only in how the 30 class centers are sampled during training:

- `global`: all centers are sampled globally from the disjoint all-nine merge.
- `radius_mix`: each episode independently uses radius 2, radius 3, or global, with
  equal probability.
- `close_only`: each episode independently uses radius 2 or radius 3, with equal
  probability.

All other training settings follow the 2026-08-07 final-core protocol: the same
immutable split-seed-0 graph artifact, 30-way 3-shot/4-query neighbor matching,
batch size 4, two-hop `9,9` context sampling, architecture, optimizer, and 2,500-update
budget.

Each arm/seed was validation-selected across checkpoints 100, 300, 900, and 2,500.
Selection maximized the macro mean over radius-2, radius-3, and global validation
panels. The frozen checkpoint was then tested once on those panels plus the secondary
within-source diagnostic. Each panel contains 500 deterministic batches of four
episodes, or 2,000 episodes total.

Horizontal comparisons within a panel are the causal comparison. Vertical comparisons
between panels are not model-quality comparisons: global episodes are much easier than
radius-confined episodes, plausibly in part because centers may come from different
source graphs. The within-source panel is graph-wide within one uniformly chosen source;
it is not radius-confined and was not used for checkpoint selection.

## Interpretation

The early radius-2 deficit visible at steps 100 and 300 was mostly transient:
`radius_mix` caught the global control by step 2,500 on both close panels. However,
neither radius treatment produced the hypothesized improvement in local
discrimination. The evidence instead favors global sampling as the best overall
treatment at this compute budget: it retains essentially the same close-panel accuracy
while performing materially better on global episodes.

With only three training seeds, small differences on the close panels should be treated
as near-ties rather than definitive rankings. The large monotonic global-panel decline
from `global` to `radius_mix` to `close_only` is consistent across all three seeds.

## Evidence

- [`figures/validation_trajectories.png`](figures/validation_trajectories.png):
  all seed-level and three-seed-mean validation accuracy/loss trajectories; a PDF
  version is stored alongside it.
- [`data/validation_trajectory.csv`](data/validation_trajectory.csv): every
  arm × seed × checkpoint × primary validation-panel cell, including loss.
- [`data/checkpoint_selections.csv`](data/checkpoint_selections.csv): frozen checkpoint
  chosen for each arm and seed.
- [`data/test_results.csv`](data/test_results.csv): per-seed frozen test results for all
  four panels.
- [`data/summary.json`](data/summary.json): strict three-seed aggregate.

The producing setup and launch/evaluation documentation live in
`scripts/experiments/setup/nm_all9_radius_finalcore/`. Raw state and logs remain on
Tucker under `/dataMeR1/phil/gfm/prodigy-radiusfc/`.
