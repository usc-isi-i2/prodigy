# What predicts GNN transfer? Nine-graph predictor study (v2)

**Status:** complete. The baseline and extended all-nine predictor sweeps have
been run, including user overlap, feature skew, one-hop aggregated features,
local computation-tree signatures, and embedding-derived topic mixtures.

## Question and design

For each target graph, exclude self-transfer and rank its eight possible source
graphs by NM transfer ROC-AUC. A pairwise predictor is good when its source
ranking agrees with that transfer ranking. The headline statistic is mean
within-target Spearman rho over the nine targets. This avoids pooling 72
dependent directed cells and removes target difficulty. Graph-label permutation
tests jointly relabel rows and columns; they never pretend matrix cells are IID.

Donor selection reports how often the metric chooses the best foreign source,
mean ROC-AUC regret from the oracle donor, and selected-donor rank. Accuracy is
included as a robustness outcome. All results use the committed complete matrix
and exclude the diagonal.

## Result 1: feature shift predicts source ranking; out-degree shift does not

| pairwise distance | mean target rho, AUC | rho, accuracy | graph-permutation p | best donor | mean AUC regret |
|---|---:|---:|---:|---:|---:|
| feature proxy-A distance | **−0.755** | −0.746 | 0.0007 | 5/9 | 0.0153 |
| feature Frechet | −0.664 | −0.667 | 0.0024 | 5/9 | **0.0131** |
| feature MMD² | −0.648 | −0.635 | 0.0034 | 5/9 | 0.0436 |
| feature centroid cosine distance | −0.640 | −0.635 | 0.0042 | 5/9 | 0.0177 |
| in-degree KS | −0.542 | −0.550 | 0.0061 | 2/9 | 0.0615 |
| out-degree KS | −0.204 | −0.177 | 0.3891 | 2/9 | 0.1005 |

Every target has a negative proxy-A rho. Thus, among the distances we have now,
**raw feature-domain separability is the clearest predictor of transfer**.
Degree distributions add a weaker structural signal, concentrated in in-degree.

## Result 2: source quality/scale is at least as important as similarity

Treating each per-graph scalar as a donor property gives:

| source property | mean target rho | permutation p | best donor | mean regret |
|---|---:|---:|---:|---:|
| feature effective dimension | **+0.780** | 0.0088 | 7/9 | 0.00049 |
| node count | +0.743 | 0.0150 | **7/9** | **0.00049** |
| degree assortativity | +0.733 | 0.0203 | 0/9 | 0.0407 |
| edge count | +0.481 | 0.1610 | 7/9 | 0.00049 |
| feature homophily | −0.460 | 0.1802 | 4/9 | 0.00348 |

The largest available foreign graph is the best AUC donor for 7/9 targets and
has essentially zero regret. It is also best by accuracy for 9/9 targets. Node
count, edge count, and feature effective dimension largely select the same
COVID/Ukraine donors, so they are not independent explanations. The safe claim
is not “size causes transfer”; it is that **source scale/coverage is a dominant
baseline that any similarity model must beat**.

Size *difference* is not a sensible distance here: absolute node-count gap has
positive rho (+0.402), meaning a more different-sized pair often transfers
better because the source is larger. A predictive model therefore needs
separate source-quality and source–target-compatibility terms.

## Result 3: raw homophily is not a pairwise compatibility predictor

Absolute feature-homophily gap has mean rho **+0.003** (p=0.996): effectively
none. The random-pair feature-similarity gap is more predictive (rho −0.497),
which implies the raw homophily scalar mixes feature-cloud geometry with actual
edge/feature coupling. Future work should use homophily lift over the random
baseline and message-passing-aware feature distributions, not raw homophily
alone. Label homophily exists for only 5/9 graphs and must not be imputed.

## Result 4: directionality exposes large but confounded source effects

Across the 36 unordered graph pairs, correlate `property(A)-property(B)` with
`AUC(A→B)-AUC(B→A)`. The largest associations are edge bio coverage (rho −0.904),
missing-bio rate (+0.890), in-degree Gini (+0.844), feature norm (+0.811), and
feature effective dimension (+0.714). Raw feature homophily is −0.612. These are
valuable hypotheses, but with nine graphs they are strongly entangled donor
signatures, not isolated causal effects.

## Can AUC on one graph predict AUC on another?

**It predicts relative donor quality better than absolute AUC.** Across all 36
pairs of targets, the seven common foreign donors have median Spearman rank
correlation **0.929** (mean 0.896; minimum 0.714). Thus, a source that ranks well
on one target usually ranks well on another.

But copying one observed foreign AUC as the prediction for another target gives
only Pearson 0.380 and **MAE 0.101 AUC** across all mutually distinct
source/reference/target triples. Averaging a source's other seven foreign AUCs
improves held-out-target MAE to 0.076, still too coarse for a precise forecast.
Self/in-domain AUC is especially weak: its Spearman correlation with mean
foreign AUC is only 0.233, with identity MAE 0.180.

If other sources have already been evaluated on the desired target, a
leave-one-cell-out additive `source effect + target effect` model reaches MAE
0.032 and R² 0.847. That is useful matrix completion, but it is **not zero-shot
target prediction**, because it uses target-specific AUC observations.

Therefore we can credibly say “performance on another graph identifies broadly
strong donors,” but not yet “this observed AUC determines the new graph's AUC.”
Calibrated forecasting needs both a donor-quality term and target descriptors.

## What the literature says to test next

The closest work supports a two-part account: transfer depends on both node
attribute shift and the distribution of local structures seen by message
passing.

- Zhu et al.'s [Ego-Graph Information maximization](https://proceedings.neurips.cc/paper_files/paper/2021/hash/0dd6049f5fa537d41753be6d37859430-Abstract.html)
  connects direct-transfer error to differences in local graph Laplacians.
- Wu et al.'s [GRADE / Non-IID Transfer Learning on Graphs](https://ojs.aaai.org/index.php/AAAI/article/download/26231/26003)
  builds a transfer bound around Weisfeiler–Lehman subtree discrepancy.
- Fang et al.'s [attribute-driven graph domain adaptation](https://proceedings.iclr.cc/paper_files/paper/2025/hash/4c802fa246c3bbcbd18e78b30bae86ca-Abstract-Conference.html)
  argues theoretically and empirically that attribute shift can exceed topology
  shift—consistent with proxy A beating degree KS here.
- Liu et al.'s [Structural Re-weighting](https://proceedings.mlr.press/v202/liu23u.html)
  distinguishes attribute shift from conditional structure shift, warning that
  marginal alignment alone is insufficient.
- Zhao et al.'s [multi-source adaptation with transferability modeling](https://arxiv.org/abs/2406.10425)
  explicitly learns graph/source selectors, the natural extension once this
  study has more than nine domains.

This motivated the extended sweep of **distribution distances over center and
sampled-neighbor features**, lightweight local-structure signatures, user
overlap, and embedding-derived topic composition reported below.

## Predictor inventory

| predictor family | status | current evidence / decision |
|---|---|---|
| user overlap | measured where comparable | Twitter pairs; Facebook pairs NA; no predictive signal |
| in/out degree distribution | complete 9/9 | in-degree moderate; out-degree weak |
| raw node-feature distances | complete 9/9 | strongest pairwise family |
| mean / left-right feature skew | complete 9/9 | aligned skew L1 is moderate; skew-distribution Wasserstein is weak |
| sampled-neighbor features | complete 9/9 | projected Fréchet is the top individual rank predictor |
| center + neighbor features | complete 9/9 | proxy A is the best practical donor selector |
| node / edge count | complete 9/9 | dominant donor baseline, not pair similarity |
| feature homophily | complete 9/9 | raw pair gap has no signal |
| label homophily | partial 5/9 | report only on observed subset |
| feature separability / proxy A | complete 9/9 | best current pairwise predictor |
| time distribution | missing/incomparable | requires raw-source timestamp extraction |
| local computation-tree signature | complete 9/9 | degree-moment/histogram distances are weak |
| full WL / ego-Laplacian distance | missing | lower priority after weak lightweight structure result |
| fixed-encoder layer representation distance | missing | tests model-specific rather than raw-data shift |
| embedding topic composition | complete 9/9 | shared 64-cluster JS distance is competitive |

## Extended candidate sweep: measured ranking

The Tucker run sampled 2,000 feature-bearing centers per graph, used undirected
fanout 100, and projected all graphs through the same 64-dimensional PCA. The
priority score is absolute within-target rho × target coverage ×
`(1 − graph-permutation p)`.

| priority | candidate | mean target rho | p | donor hits | regret | decision |
|---:|---|---:|---:|---:|---:|---|
| 1 | neighbor-mean projected Fréchet | **−0.767** | 0.0005 | 6/9 | 0.0431 | strongest ranking association, imperfect selector |
| 2 | raw feature proxy A | −0.755 | 0.0005 | 5/9 | 0.0153 | retain as simple baseline |
| 3 | center+neighbor projected Fréchet | −0.704 | 0.0035 | 5/9 | 0.0341 | secondary |
| 4 | center+neighbor proxy A | −0.696 | 0.0020 | **7/9** | **0.0126** | best operational compatibility metric |
| 5 | center+neighbor MMD² | −0.683 | 0.0030 | 4/9 | 0.0345 | redundant with stronger metrics |
| 6 | embedding-topic JS | −0.656 | 0.0085 | 5/9 | 0.0158 | retain as content control |
| 7 | center+neighbor aligned skew L1 | −0.630 | 0.0275 | 5/9 | 0.0339 | exploratory |

Local computation-tree proxy A is weaker (rho −0.382, p=0.158), and its
MMD/Fréchet variants are weaker still. User Jaccard is null (rho −0.003,
p=0.996); directional containment is also near zero. Keep user overlap as a
leakage/continuity control, not a primary predictor.

### Harder combined-model test

For each held-out graph `g`, fitting excludes every transfer pair having `g` as
either source or target. The model then predicts all eight `source→g` cells.

| model | mean target rho | MAE | R² | donor hits |
|---|---:|---:|---:|---:|
| source-only ridge | **0.952** | 0.078 | 0.246 | **7/9** |
| all full-coverage features, Extra Trees | 0.929 | 0.064 | 0.413 | 4/9 |
| source+target graph descriptors, Extra Trees | 0.913 | **0.059** | **0.497** | 2/9 |
| original six distances, ridge | 0.717 | 0.113 | −0.579 | 4/9 |

**Donor ranking is dominated by universal source strength**; source-only
descriptors already recover it. **Absolute AUC needs target calibration**;
source+target descriptors halve error relative to copying one AUC from another
graph (0.059 versus 0.101). The complete 109-feature model does not improve
further, so pairwise association has not yet become incremental out-of-graph
predictive value.

### Final-core three-seed rerun: independent confirmation of donor strength

The later final-core protocol retrained every specialist with three seeds,
exactly 2,500 balanced-source updates, validation-only checkpoint selection,
and one 500-episode fixed held-out test per seed. Its test score is episodic NM
accuracy, not ROC-AUC, so this is a robustness comparison rather than a
replacement 9×9 matrix.

The nine specialists' new mean test scores correlate **rho = 0.800** with their
historical mean foreign-transfer ROC-AUC. The ranking is stable at the top:
COVID (0.579), Ukraine/Russia (0.447), suspended Ukraine/Russia (0.366), and
TwiBot-20 (0.333). Across-source seed standard deviations are small
(0.0007–0.0032) except Facebook (0.0132), whose selected checkpoints also vary.

Feature effective dimension remains the strongest measured source descriptor
(rho = 0.800 with the new score). Node count falls from rho 0.743 in the old
matrix analysis to **0.400** under fixed exposure, while edge count is 0.467.
Thus the original “large sources win” result was partly exposure/coverage
confounding, but a reproducible source-quality axis remains after controlling
the optimizer budget.

This closes the training-seed and checkpoint-selection gaps for universal
specialist quality. It does **not** close the per-target replication gap: the
rerun evaluates a shared all-nine held-out test distribution rather than nine
separate target columns.

### Revised priority

1. Make source-only strength the mandatory donor-ranking baseline.
2. Use center+neighbor proxy A as the primary compatibility metric and
   neighbor-mean Fréchet as its ranking-oriented companion.
3. Retain embedding-topic JS and aligned skew L1 as low-cost content controls.
4. Test incremental value with graph-level holdout, never random matrix cells.
5. Keep user overlap as a leakage diagnostic; deprioritize the current
   degree-only computation-tree metrics.
6. Next expensive candidate: fixed-checkpoint layer representations. Temporal
   distance remains partial because timestamps are not comparable across all
   nine final graph artifacts.

## Reproducibility and limitations

- Branch/worktree and Tucker commands are in the paired setup README.
- Machine-readable tables and the availability ledger are in `data/`.
- The checked-in permutation run uses 9,999 seeded graph-label permutations.
- There are only nine domains, many scalars identify the same large corpora,
  and the transfer matrix has one realization per source–target pair. These are
  ranking results and hypothesis generators, not causal estimates.
