# Bias-only LP rerun

Completed 18 fresh source models and 162 evaluation cells. All runs stopped on source-validation plateau; none reached the 100k safety cap. Bias is trained jointly with the encoder, initialized at -log(5), and frozen for every transfer target. No extra scale or target calibration in headline scores. Seed 0, same uniform 1:5 evaluation pairs as the previous matrices, fixed ten-neighbor disjoint context. FR unchanged.

| Model | Subset | Previous AUC | Bias AUC | Previous BCE | Bias BCE |
|---|---|---:|---:|---:|---:|
| node_bias | all | 0.6926 | 0.7364 | 0.7425 | 0.5338 |
| node_bias | same_graph | 0.9319 | 0.9691 | 0.6249 | 0.1355 |
| node_bias | cross_graph | 0.6627 | 0.7073 | 0.7572 | 0.5836 |
| node_neighbors_bias | all | 0.8223 | 0.8880 | 0.7841 | 0.4293 |
| node_neighbors_bias | same_graph | 0.9543 | 0.9782 | 0.6168 | 0.1120 |
| node_neighbors_bias | cross_graph | 0.8059 | 0.8767 | 0.8050 | 0.4690 |

Constant p=1/6 BCE is 0.4506. All 18 same-graph cells beat it. Across the 72 cross-graph cells, 16 node-only and 45 neighbor cells beat it. Bias improves average cross-graph ranking and BCE, but target probability calibration remains imperfect: mean cross-graph BCE is still above the constant baseline for both variants. Worst neighbor BCE is Facebook→Election 2020 at 2.2000. AUC improvement reflects altered representation learning, because adding a constant bias to fixed logits cannot change their ranking. One seed; no uncertainty interval or multi-seed robustness claim.

The standalone Midterm bias-only experiment is reproduced: AUC 0.98224 and BCE 0.10782. Selected checkpoints span 6k–54k updates; final stopping steps span 12k–60k.

Validation: all 162 cells checked for exact equality of test pairs/masks against the previous uniform evaluation, frozen bias application, finite scores, 1:5 class counts and recomputed BCE. Known-edge rejection also asserted during evaluation.

Training/evaluation revision fc6231c; branch codex/bias-lp-matrices, local worktree /tmp/mixture-bias-lp. Tucker worktree /dataMeR1/phil/gfm/mixture-scaling-bias-lp; state/bias_s0. Launch and reproduction protocol: docs/bias_lp_matrices.md. W&B histories remain offline. Previous gallery: gallery_uniform_raw.html; previous matrices and FR results remain intact.
