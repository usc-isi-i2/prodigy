# What predicts GNN transfer? Nine-graph predictor study (v2)

**Status:** the complete 9×9 analysis of already available predictors is done.
The Tucker extractor for user overlap, feature skew, and one-hop aggregated
feature distributions is implemented but has not yet produced its artifact.

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
| feature proxy-A distance | **−0.755** | −0.746 | 0.0005 | 5/9 | 0.0153 |
| feature Frechet | −0.664 | −0.667 | 0.0020 | 5/9 | **0.0131** |
| feature MMD² | −0.648 | −0.635 | 0.0055 | 5/9 | 0.0436 |
| feature centroid cosine distance | −0.640 | −0.635 | 0.0030 | 5/9 | 0.0177 |
| in-degree KS | −0.542 | −0.550 | 0.0060 | 2/9 | 0.0615 |
| out-degree KS | −0.204 | −0.177 | 0.3755 | 2/9 | 0.1005 |

Every target has a negative proxy-A rho. Thus, among the distances we have now,
**raw feature-domain separability is the clearest predictor of transfer**.
Degree distributions add a weaker structural signal, concentrated in in-degree.

## Result 2: source quality/scale is at least as important as similarity

Treating each per-graph scalar as a donor property gives:

| source property | mean target rho | permutation p | best donor | mean regret |
|---|---:|---:|---:|---:|
| feature effective dimension | **+0.780** | 0.0095 | 7/9 | 0.00049 |
| node count | +0.743 | 0.0200 | **7/9** | **0.00049** |
| degree assortativity | +0.733 | 0.0205 | 0/9 | 0.0407 |
| edge count | +0.481 | 0.1655 | 7/9 | 0.00049 |
| feature homophily | −0.460 | 0.1680 | 4/9 | 0.00348 |

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

Absolute feature-homophily gap has mean rho **+0.003** (p=0.995): effectively
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

The highest-priority new metric is therefore a **distribution distance over
`[center feature, sampled-neighbor mean]`**, followed by a WL/ego-Laplacian
distance. User overlap is an important leakage/continuity control; temporal
distance and language/topic composition are plausible social-corpus controls.

## Predictor inventory

| predictor family | status | current evidence / decision |
|---|---|---|
| user overlap | missing | extractor ready for Twitter pairs; Facebook pairs NA |
| in/out degree distribution | complete 9/9 | in-degree moderate; out-degree weak |
| raw node-feature distances | complete 9/9 | strongest pairwise family |
| mean / left-right feature skew | missing | extractor ready; test aligned-skew L1 and distributional skew distance |
| sampled-neighbor features | partial 8/9 means | existing mean distances were weak; v2 tests PAD/MMD/projected Frechet on 9/9 |
| center + neighbor features | partial 8/9 means | same; v2 extractor ready |
| node / edge count | complete 9/9 | dominant donor baseline, not pair similarity |
| feature homophily | complete 9/9 | raw pair gap has no signal |
| label homophily | partial 5/9 | report only on observed subset |
| feature separability / proxy A | complete 9/9 | best current pairwise predictor |
| time distribution | missing/incomparable | requires raw-source timestamp extraction |
| WL / computation-tree / ego-Laplacian distance | missing | highest-priority structural addition |
| fixed-encoder layer representation distance | missing | tests model-specific rather than raw-data shift |
| language/topic composition | missing | useful social-domain control |

## Reproducibility and limitations

- Branch/worktree and Tucker commands are in the paired setup README.
- Machine-readable tables and the availability ledger are in `data/`.
- The checked-in permutation run uses 1,999 seeded graph-label permutations.
- There are only nine domains, many scalars identify the same large corpora,
  and the transfer matrix has one realization per source–target pair. These are
  ranking results and hypothesis generators, not causal estimates.
