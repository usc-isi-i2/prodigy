# What predicts GNN transfer? Nine-graph predictor study (v2)

**Status:** complete, including the proper final-core ROC-AUC extension. The
baseline and extended all-nine predictor sweeps have been run, then tested on a
strict final-core matrix: 9 sources × 9 targets × 3 training seeds = 243
fixed-test specialist cells with accuracy, macro-F1, and multiclass ROC-AUC.

## Question and design

For each target graph, exclude self-transfer and rank its eight possible source
graphs. The historical outcome is NM transfer ROC-AUC; the authoritative
final-core outcome is three-seed mean 30-way one-vs-rest macro ROC-AUC, with
accuracy retained as a companion robustness outcome. A pairwise predictor is
good when its source ranking agrees with the transfer ranking. The headline
statistic is mean within-target Spearman rho over the nine targets. This avoids
pooling 72 dependent directed cells and removes target difficulty. Graph-label
permutation tests jointly relabel rows and columns; they never pretend matrix
cells are IID.

Donor selection reports how often the metric chooses the best foreign source,
mean regret from the oracle donor, and selected-donor rank. Both matrices are
complete and all headline analyses exclude the diagonal. The final-core matrix
is the stronger evidence because it fixes the optimizer budget, checkpoint,
target episode stream, and repeats training across three seeds.

## Historical result 1: feature shift predicts source ranking; out-degree shift does not

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

## Historical result 2: source quality/scale is at least as important as similarity

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

## Historical result 3: raw homophily is not a pairwise compatibility predictor

Absolute feature-homophily gap has mean rho **+0.003** (p=0.996): effectively
none. The random-pair feature-similarity gap is more predictive (rho −0.497),
which implies the raw homophily scalar mixes feature-cloud geometry with actual
edge/feature coupling. Future work should use homophily lift over the random
baseline and message-passing-aware feature distributions, not raw homophily
alone. Label homophily exists for only 5/9 graphs and must not be imputed.

## Historical result 4: directionality exposes large but confounded source effects

Across the 36 unordered graph pairs, correlate `property(A)-property(B)` with
`AUC(A→B)-AUC(B→A)`. The largest associations are edge bio coverage (rho −0.904),
missing-bio rate (+0.890), in-degree Gini (+0.844), feature norm (+0.811), and
feature effective dimension (+0.714). Raw feature homophily is −0.612. These are
valuable hypotheses, but with nine graphs they are strongly entangled donor
signatures, not isolated causal effects.

## Can AUC on one graph predict AUC on another?

**It predicts relative donor quality better than absolute AUC.** On the proper
final-core matrix, the seven common foreign donors across each pair of targets
have median Spearman rank correlation **0.821** (mean 0.773; minimum 0.429).
This is weaker than the historical matrix's median 0.929, but the qualitative
result survives: broadly strong donors tend to remain strong.

Copying one observed foreign AUC as the prediction for another target gives
Pearson 0.328, Spearman 0.307, and **MAE 0.068 AUC** across all 504 mutually
distinct source/reference/target triples. Averaging a source's other seven
foreign AUCs improves held-out-target MAE to 0.048. Self/in-domain AUC is still
not a calibrated forecast: Spearman with mean foreign AUC is 0.450 and identity
MAE is 0.101.

If other sources have already been evaluated on the desired target, a
leave-one-cell-out additive `source effect + target effect` model reaches MAE
0.027 and R² 0.770. That is useful matrix completion, but it is **not zero-shot
target prediction**, because it uses target-specific AUC observations.

Therefore we can credibly say “performance on another graph identifies broadly
strong donors,” but not “this observed AUC determines the new graph's AUC.” A
zero-shot forecast still needs a donor-quality term and target descriptors.

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
| raw node-feature distances | complete 9/9 | proxy A is the top proper-AUC rank predictor |
| mean / left-right feature skew | complete 9/9 | aligned skew L1 is moderate; skew-distribution Wasserstein is weak |
| sampled-neighbor features | complete 9/9 | projected Fréchet is a close second for proper AUC |
| center + neighbor features | complete 9/9 | proxy A has the lowest proper-AUC regret among tested pair metrics |
| node / edge count | complete 9/9 | dominant donor baseline, not pair similarity |
| feature homophily | complete 9/9 | raw pair gap has no signal |
| label homophily | partial 5/9 | report only on observed subset |
| feature separability / proxy A | complete 9/9 | best current proper-AUC pairwise predictor |
| time distribution | missing/incomparable | requires raw-source timestamp extraction |
| local computation-tree signature | complete 9/9 | degree-moment/histogram distances are weak |
| full WL / ego-Laplacian distance | missing | lower priority after weak lightweight structure result |
| fixed-encoder layer representation distance | missing | tests model-specific rather than raw-data shift |
| embedding topic composition | complete 9/9 | shared 64-cluster JS distance is competitive |

## Historical extended candidate sweep: measured ranking

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

## Final-core three-seed 9×9 proper-AUC replication

The authoritative rerun uses every final-core specialist trained with seeds
0/1/2 for exactly 2,500 balanced-source updates and evaluates only the terminal
step-2,500 checkpoint. Every source/seed checkpoint is evaluated on each of the
nine targets using 512 fixed `static_test` episodes, with message passing
restricted to `static_train`. This is a strict **243-cell specialist matrix**.

The new metric contract preserves accuracy, macro-F1, and multiclass one-vs-rest
macro ROC-AUC from the same logits. It also verifies each observed episode
stream against the published fixed-grid fingerprint ledger. The proper AUC
matrix is therefore directly comparable across sources without relabeling the
accuracy field.

### The historical AUC ranking replicates, but not perfectly

- Mean within-target correlation between historical and final-core AUC donor
  ranks: **rho = 0.854**.
- Same best donor in **5/9** targets.
- Overall correlation across the 72 foreign cells: **rho = 0.827**.
- Correlation of sources' mean foreign AUC: **rho = 0.933**.
- Mean cross-seed within-target donor-rank stability: **rho = 0.944**.

The result is a strong replication of broad donor quality and within-target
ordering, though weaker than the companion accuracy comparison (rho 0.934 and
7/9 best-donor agreement). The metric choice changes some close donor rankings;
it does not erase the transfer structure.

### Predictor ranking against proper final-core AUC

| priority | candidate | mean target rho | seed rhos (0/1/2) | p | donor hits | AUC regret |
|---:|---|---:|---|---:|---:|---:|
| 1 | raw feature proxy A | **−0.738** | −0.735/−0.725/−0.694 | 0.0006 | 4/9 | 0.0141 |
| 2 | neighbor-mean projected Fréchet | −0.733 | −0.730/−0.722/−0.749 | 0.0006 | **5/9** | 0.0278 |
| 3 | center+neighbor projected Fréchet | −0.693 | −0.680/−0.664/−0.733 | 0.0018 | **5/9** | 0.0208 |
| 4 | embedding-topic JS | −0.672 | −0.669/−0.635/−0.672 | 0.0017 | **5/9** | 0.0082 |
| 5 | raw feature Fréchet | −0.661 | −0.640/−0.632/−0.622 | 0.0012 | 4/9 | 0.0110 |
| 6 | center+neighbor proxy A | −0.660 | −0.654/−0.626/−0.656 | 0.0016 | **5/9** | **0.0080** |

Raw feature proxy A, not neighbor Fréchet, is the top proper-AUC rank
predictor. The two are close and stable across seeds. Center+neighbor proxy A
remains the lowest-regret pairwise selector, while topic JS is nearly tied.

The negative controls also hold. In-degree KS is moderate (rho −0.434,
p=0.0379), out-degree KS is weak (−0.175, p=0.4488), the lightweight local
computation-tree proxy A is weak (−0.254, p=0.3448), and user Jaccard is null
(−0.051, p=0.887; Twitter-only coverage). Raw feature-homophily gap is also null
(+0.098, p=0.670), whereas the random-pair feature-similarity gap remains
informative (−0.484, p=0.0131).

Source strength survives fixed exposure and proper AUC: feature effective
dimension has rho **+0.735** (p=0.0103), in-degree maximum +0.706 (p=0.0173),
node count +0.677 (p=0.0242), and edge count +0.611 (p=0.0433). These correlated
descriptors remain a mandatory baseline, not separate causal explanations.

### Hard graph-holdout prediction against proper AUC

For each held-out graph, training excludes every pair having that graph as
source or target. No model sees any transfer result involving the test graph.

| model | mean target rho | MAE AUC | R² | donor hits |
|---|---:|---:|---:|---:|
| source + original distances, ridge | **0.841** | 0.0616 | −0.158 | **6/9** |
| source-only Extra Trees | 0.828 | 0.0484 | 0.232 | 5/9 |
| all full-coverage features, Extra Trees | 0.802 | 0.0488 | 0.327 | 3/9 |
| all full-coverage features, ridge | 0.759 | 0.0473 | 0.320 | 3/9 |
| handpicked source + compatibility, Extra Trees | 0.746 | **0.0470** | **0.394** | 4/9 |

No combined model dominates. Adding the original distances to ridge improves
ranking and donor hits but damages absolute calibration. The handpicked model
slightly improves MAE and R² over source-only Extra Trees but worsens ranking
and donor hits. Thus pairwise distances have strong univariate association, but
their **incremental zero-shot value is modest and metric-dependent**. The
accuracy companion result was more favorable to the handpicked combination;
that claim should not be generalized to AUC.

### Revised priority

1. Make source-only strength the mandatory donor-ranking and forecasting
   baseline.
2. Use raw feature proxy A as the primary proper-AUC compatibility metric;
   retain neighbor-mean Fréchet as its message-passing-aware companion.
3. Use center+neighbor proxy A when low selection regret matters, and retain
   embedding-topic JS as a low-cost content control.
4. Test incremental value with graph-level holdout, never random matrix cells;
   report ranking and calibration separately because they select different
   models here.
5. Keep user overlap as a leakage diagnostic; deprioritize the current
   degree-only computation-tree metrics.
6. Next expensive candidate: fixed-checkpoint layer representations. Temporal
   distance remains missing because timestamps are not comparable across all
   nine final graph artifacts.

## Reproducibility and limitations

- Branch/worktree and Tucker commands are in the paired setup README.
- Raw final-core AUC evidence, imported matrices, predictor rankings, graph-
  holdout predictions, and the availability ledger are in `data/final_core_auc/`
  and `data/`.
- The checked-in permutation run uses 9,999 seeded graph-label permutations.
- Partial-coverage permutation tests retain their finite source/target mask;
  unavailable Facebook↔Twitter identity overlap is never converted to zero.
- There are only nine domains and many scalars identify the same large corpora.
  The final-core matrix has three independent training seeds but deliberately
  reuses one fixed 512-episode stream per target across every cell. These are
  robust ranking results and hypothesis generators, not causal estimates or an
  eval-episode confidence interval.
