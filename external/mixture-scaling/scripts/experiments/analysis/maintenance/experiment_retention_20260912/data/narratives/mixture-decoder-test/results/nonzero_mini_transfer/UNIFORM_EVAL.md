# Uniform LP evaluation

Frozen checkpoints, evaluated using independently uniform negative endpoints, excluding self-loops and every known undirected edge. Both LP variants use identical pairs. All 2,000 original positives and their calibration/test assignment are preserved. Calibration: 600 positives and 3,000 negatives; final test: 1,400 positives and 7,000 negatives. Training and checkpoint selection are unchanged. FR is unchanged.

| Macro average over 81 cells | AUC | Raw BCE | Balanced-subset BCE | Target-calibrated BCE |
|---|---:|---:|---:|---:|
| Node only | 0.6926 | 0.7425 | 0.6493 | 0.3916 |
| Node + 10 neighbors, disjoint context | 0.8223 | 0.7841 | 0.6144 | 0.3283 |

The constant p=1/6 baseline has BCE 0.4506 at this class ratio: raw model probabilities remain poorly calibrated despite useful ranking. Target-calibrated scores use labeled target calibration pairs and are not zero-shot results. Uniform negatives are easier than the previous degree-matched negatives; increases in AUC are protocol changes, not improved checkpoints. Historical balanced and current 1:5 raw BCE are not directly comparable. Balanced-subset BCE uses all test positives and a fixed equally sized subset of uniform negatives.

All 162 evaluations completed. Pair equality between model variants, positive preservation, class counts and finite scores were checked locally; exact known-edge rejection is asserted by the evaluator. Code: codex/uniform-mini-eval, evaluator revision 8f08e16. Tucker output: /dataMeR1/phil/gfm/mixture-scaling-uniform-eval/state/uniform_eval_s0_v2.

Previous degree-matched results remain in gallery_degree_matched.html and their original result directories. Existing W&B matrix runs describe those historical evaluations; these uniform results have not been uploaded.
