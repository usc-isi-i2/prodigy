# Corrected disjoint-context neighbor LP

Nine corrected models and all 81 transfer cells are complete. All models met
the existing validation-plateau criterion (10k–86k updates); none hit 100k.
The actual cached neighbor samples contain only context edges: zero overlap
with supervised positive edges, validation edges or final test edges in all
nine graphs. The original 70/15/15 edge partition is unchanged. Evaluation
endpoints, labels, and calibration/test masks match the original experiment
exactly in every cell. See `node_neighbors_disjoint/disjoint_audit.json`.

The original 70% training pool is now split equally into fixed context edges
and positive supervision edges. Fixed up-to-10-neighbor means use only context.
Both graphs and feature vectors are unchanged. Exact negatives still exclude
ALL known edges. This reduces context density and training-positive coverage,
so it is not a clean causal estimate of shortcut removal alone.

## Train–validation gaps at selected checkpoints

Mean BCE gap fell from 0.03151 to 0.00511 (83.8% reduction).
China/HK: 0.0791→0.0073; Facebook: 0.0587→0.0126; Midterm: 0.0472→0.0028;
COVID Political: 0.0502→0.0084. Some residual overfitting remains. These are
100-update mean training losses versus fixed source-validation losses, evaluated
at each model's selected checkpoint, not matched optimizer-update budgets.
The complete per-source comparison is `disjoint_comparison.csv`.

## Held-out LP performance

| Scope | Original AUC | Corrected AUC | Original raw BCE | Corrected raw BCE |
|---|---:|---:|---:|---:|
| All 81 cells | 0.6550 | 0.6554 | 0.8975 | 0.9242 |
| 72 cross-graph cells | 0.6459 | 0.6416 | 0.8678 | 0.8810 |
| 9 same-graph cells | 0.7275 | 0.7655 | 1.1355 | 1.2698 |

The correction substantially closes the training/validation gap; it does not
solve probability calibration or uniformly improve transfer. Same-graph AUC
improves on 8/9 datasets, while cross-graph average AUC decreases slightly.
Raw balanced-test BCE remains poor. Do not compare it directly to the 1:5
train/validation BCE, and do not infer seed-level statistical significance.

## Artifacts and provenance

`gallery.html` now presents corrected neighbor curves and AUC/BCE matrices,
with a per-source original-versus-corrected table. `gallery_original.html`
preserves the earlier gallery. Node-only FR and LP results are unchanged.
Raw corrected metrics/scores, summaries, histories and audit are in
`node_neighbors_disjoint/`; images/PDFs use `node_neighbors_disjoint_*`.

Training revision: `8f7f90e`, branch `codex/disjoint-neighbor-lp`, Tucker
worktree `/dataMeR1/phil/gfm/mixture-scaling-disjoint-neighbor`, state root
`state/disjoint_context_s0`. Four tests passed locally and in production;
20-step end-to-end smoke results are separate from the full experiment.

[Corrected W&B matrix](https://wandb.ai/eibl-usc/nonzero-mini-transfer/runs/mthmn6a5):
all nine training histories and all 81 evaluation cells uploaded.
