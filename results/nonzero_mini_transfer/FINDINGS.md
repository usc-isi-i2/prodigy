> LP evaluation has since changed to uniform negatives at 1:5. See [UNIFORM_EVAL.md](UNIFORM_EVAL.md) for current matrices. Numbers below retain the historical evaluation protocol.

# Nonzero-mini transfer matrices

**Update:** The overlapping-context neighbor run below is historical. The active
gallery uses the [corrected disjoint-context retraining](DISJOINT_CONTEXT.md).

Requested sequence: node-only feature reconstruction (FR), node-only LP, then
node + fixed-10-neighbor LP. Each stage is nine single-source seed-0 models
against all nine targets. Graphs use `nonzero_features_v1`, with 500k-node
one-hop-walk seed-0 induced Ukraine and COVID minis. Other graphs are full
nonzero views. Input features are unchanged beyond filtering/selection.

## Feature reconstruction: complete

All 9 models stopped under the prescribed validation-plateau rule (14k–66k
updates); none reached the 100k safety cap. This is operational convergence:
three checks without a cumulative absolute 1e-4 improvement, checks every 2k.
The best checkpoint is selected by source validation only. All 81 cells are
in `fr/feature_reconstruction_matrix.csv`, with per-cell provenance and ten
mask-replicate metrics under `fr/fp/` and training summaries in `fr/summaries.json`.

The objective is squared cosine error on the masked half of input coordinates,
not BCE. Lower is better. Node partitions are disjoint within each graph:
70% training, 15% early-stopping validation, 15% final test. Validation uses
at most 20,480 nodes; final evaluation uses every test node and ten mask seeds.
Models share the same evaluation nodes and mask seeds. Standard deviation across
masks is not uncertainty across training seeds. Cross-graph identities/features
have not been deduplicated, so graph-local separation is not a claim of global
identity separation.

The same-source model has the lowest error on 8/9 targets. On Ukraine, COVID
training obtains 0.0197 versus Ukraine training's 0.0206. COVID training has the
lowest equal-target macro error (0.01957). This is one training seed, and these
absolute errors do not establish how much improvement comes from contextual
reconstruction without a constant-feature baseline.

[W&B FR matrix](https://wandb.ai/eibl-usc/nonzero-mini-transfer/runs/2k03zro1)
contains the 81-cell table and evaluation artifacts. Nine training runs in the
same project contain train/validation curves and per-target summary metrics.

## LP stages: complete

Both 81-cell matrices are complete. All 18 LP models stopped under the same
operational plateau criterion; none reached the safety cap. Node-only models
used 16k–72k updates; fixed-neighbor models used 8k–56k. Every evaluation cell has
1,400 positive and 1,400 negative final-test pairs; 600 of each were reserved for
calibration. The exact endpoints, labels and calibration/test masks match between
the two models in all 81 cells, and all holdout-leakage checks are zero.

| Scope | Node AUC | Node + 10-neighbor AUC | Node BCE | Node + 10-neighbor BCE |
|---|---:|---:|---:|---:|
| All 81 cells | 0.6110 | 0.6550 | 0.8181 | 0.8975 |
| 72 cross-graph cells | 0.5948 | 0.6459 | 0.7671 | 0.8678 |
| 9 same-graph cells | 0.7409 | 0.7275 | 1.2260 | 1.1355 |

Neighbor context improves AUC in 66/81 cells, but improves raw BCE in only 19/81.
This is improved transfer ranking, not improved probability calibration. TwiBot20
is the strongest equal-target macro-AUC source in both LP variants (0.6596 node,
0.7157 with neighbors). Same-graph performance is mixed; Facebook's AUC falls
from 0.8574 to 0.6903 with context. These are one-seed observations, not established
significance across model initializations.

Raw BCE is materially worse than the balanced constant-p=.5 baseline (0.6931)
in many cells. China/HK node-only diagonal BCE is 2.7838, dominated by negative
BCE 5.4723; its test logits have a large positive tail. AUC 0.6407 does not imply
useful calibrated probabilities. Raw scores and separately validation-calibrated
metrics remain in the JSON/NPZ artifacts; primary tables do not hide the raw BCE.

The models have the same hidden/output widths, but concatenating neighbor means
increases encoder parameters (262,656 → 459,264). The comparison therefore does
not isolate topology from parameter count. No feature normalization was added.

[Node-only LP on W&B](https://wandb.ai/eibl-usc/nonzero-mini-transfer/runs/am8myn0a) and
[node + 10-neighbor LP on W&B](https://wandb.ai/eibl-usc/nonzero-mini-transfer/runs/ot35dkcq).
All 27 training histories and all 243 evaluation cells are uploaded; logging
remains offline by default until an explicit sync.
The full paired comparison is `lp_comparison.csv`; AUC/BCE heatmaps are under
`figures/`. The graph-local 70/15/15 unique undirected edge partition
keeps source checkpoint-validation positives distinct from final evaluation.
Uniform 1:5 training negatives exclude all known graph edges in either direction.
Final evaluation uses 1:1 degree-matched negatives; raw-dot BCE across these two
class balances must not be compared directly. See `docs/nonzero_mini_transfer.md`.

## Reproduction and provenance

FR runner: `9b54a0a`, branch `codex/mini-transfer-fast`, Tucker worktree
`/dataMeR1/phil/gfm/mixture-scaling-mini-transfer`.
LP runner: `1a851fd`, branch `codex/mini-lp-fast`, Tucker worktree
`/dataMeR1/phil/gfm/mixture-scaling-mini-lp`.
Mini materialization: PRODIGY `86535f66`, verified inventory on branch
`codex/walk-mini-pilot` at `a2bfd16f`.
Training is FP32 with TF32 off, GPU-resident features and seeded CUDA masks/pairs.
Three unit tests per runner passed locally and under production PyTorch 2.0.1;
both LP variants passed end-to-end 20-update smoke training/evaluation.
Smoke outputs are separate and are not these experiment results.
