# Consolidated ogbl-collab baseline findings

Historical baseline phase. For the subsequent sub-million-parameter experiments,
negative-panel exposure audit, and current outcome, see the
[compact-model consolidation](../../../baselines/ogbl_collab_compact_joint/CONSOLIDATED_FINDINGS.md).

## Bottom line

On the official temporal `ogbl-collab` split, author features alone predict a
substantial fraction of future collaborations, graph aggregation adds a clear gain,
and the strongest reproduced score comes from a recurrence-oriented structural
heuristic. The headline ranking is not a single controlled leaderboard, however:
AA-DC uses validation edges as graph input at test time, whereas the feature-only
and GraphSAGE campaigns do not.

The most informative controlled comparison is therefore the matched
feature-only-to-GraphSAGE contrast. A nonlinear feature encoder with cosine scoring
reaches `36.95 +/- 1.12%` test Hits@50; adding two-layer GraphSAGE over the
unique, unweighted training graph raises this to `46.63 +/- 1.22%`, a gain of `9.68`
percentage points. Only `3.26` points of that gain appear on collaborations novel
relative to training, compared with `20.79` points on recurring pairs.

## Results and comparison boundaries

All learned results below report mean and sample standard deviation over training
seeds 0--2. Raw cosine and AA-DC are deterministic.

| Method | Test Hits@50 (%) | Evidence class | Direct comparison boundary |
| --- | ---: | --- | --- |
| Raw feature cosine | 26.27 | valid | Features only; no learned parameters |
| Linear map + cosine | 34.52 +/- 0.14 | valid | Features only |
| Nonlinear MLP + cosine | 36.95 +/- 1.12 | valid | Matched reference for GraphSAGE + cosine |
| Matched GraphSAGE + cosine | 46.63 +/- 1.22 | valid with disclosed smoke exposure | Same split, scorer, and no validation-edge reuse as the nonlinear MLP |
| OGB-style GraphSAGE | 48.62 +/- 0.99 | valid | Different encoder, weighted adjacency, and edge scorer; still excludes validation edges |
| AA-DC | 68.02 | valid, protocol-distinct | Uses train + validation edges as graph input for test |

The OGB-style GraphSAGE result closely reproduces the published GraphSAGE mean of
`48.10 +/- 0.81%`; its `0.52`-point difference is not evidence of superiority.
AA-DC exactly reproduces its published `68.02%` score to reported precision, but its
`19.40`-point lead over OGB-style GraphSAGE cannot be attributed solely to the
method because their test-time graph inputs differ.

## What the experiments establish

1. **The supplied author features carry useful link signal.** Raw cosine already
   reaches `26.27%` Hits@50, and learning a nonlinear shared representation raises
   this to `36.95%` without graph access, node IDs, edge weights, or years.

2. **Training-graph message passing provides a real but recurrence-heavy gain.**
   Matched GraphSAGE improves Hits@50 from `36.95%` to `46.63%`. On test positives
   seen during training it improves `67.61%` to `88.40%`; on novel positives it
   improves `19.23%` to `22.49%`.

3. **The extra OGB-style machinery mainly improves recurring-edge prediction.**
   Moving from matched GraphSAGE to the weighted three-layer OGB-style recipe raises
   overall Hits@50 by `1.99` points and seen-in-training Hits@50 from `88.40%` to
   `93.76%`, while novel Hits@50 changes only from `22.49%` to `22.54%`.

4. **A tuned structural heuristic is extremely strong on this benchmark.** AA-DC
   reaches `68.02%` with no learned parameters, using time-decayed weighted
   Adamic--Adar, validation-calibrated gates, and train-plus-validation graph input
   for test. This demonstrates the strength of temporal recurrence and local
   structure on Collab, not a protocol-matched neural-versus-heuristic comparison.

5. **Naive endpoint concatenation is a poor feature-only scorer here.** The
   post-hoc symmetric nonlinear concat model reaches `22.27 +/- 0.59%`, well below
   shared-encoder cosine. Removing undirected symmetry does not rescue it: the
   ordered nonlinear diagnostic reaches `21.04 +/- 0.69%`. The linear concat arms
   reach only about `2.2%` because an additive linear score has no endpoint
   interaction term. These are architecture diagnostics, not preregistered members
   of the original cosine ladder.

6. **Validation selection did not materially hide a better feature-only result.**
   In the deliberately non-admissible test-oracle diagnostic, retrospective
   test-based epoch selection gives a mean nonlinear-cosine advantage of only
   `0.056` Hits@50 points. This supports the stability of the reported selection but
   must not replace the production result.

## Interpretation for subsequent work

`ogbl-collab` is a useful implementation and recurrence-prediction benchmark, but
its aggregate Hits@50 should not be treated as a clean measure of novel-link
generalization. Any claim about transferable representations or discovery of new
relationships should report the novel-versus-training stratum alongside the
official metric. Future neural-versus-heuristic comparisons should also freeze the
same test-time graph input, especially whether validation edges are included.

The present evidence does not show that GraphSAGE learns a broadly general link
mechanism: it shows a modest novel-link improvement and a much larger recurrence
improvement. Nor does AA-DC's lead establish that learning is unnecessary under a
matched input protocol.

## Evidence status and provenance

- The cosine MLP, matched GraphSAGE, OGB-style GraphSAGE, and AA-DC result cells are
  complete and have committed validation receipts.
- The symmetric concat campaign is complete but classified as post-hoc exploratory.
- Ordered concat and the test-oracle trajectory are complete diagnostic-only
  campaigns and are excluded from benchmark claims.
- Smoke artifacts are excluded. The matched GraphSAGE production result retains a
  disclosed qualification because a full-data smoke run exposed test output after
  the protocol was frozen; no scientific setting changed afterward.
- All neural campaigns use the same official split fingerprint:
  `07f7af8e654bda27caad60ed74c479f48780343826543dd613d90cb4e979f9f4`.

Source analyses and machine-readable receipts:

- [Feature-only MLP and concat findings](../../../baselines/ogbl_collab_mlp_lp/FINDINGS.md)
- [Matched and OGB-style GraphSAGE results](../../../baselines/ogbl_collab_sage_lp/RESULTS.md)
- [AA-DC reproduction](../../../baselines/ogbl_collab_aadc/RESULTS.md)
