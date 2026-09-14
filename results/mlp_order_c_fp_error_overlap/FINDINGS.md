# FP example-level agreement across Order C node-only MLPs

## Question

Do node-only MLPs pretrained on different source graphs find the same target nodes difficult under feature prediction (FP)?

## Protocol

- Retrained the nine single-source Order C specialists at seed 0 with the repaired node-only transfer protocol.
- Evaluated every source model on the same validation nodes for each target (10,000 nodes, except `ukr_rus_suspended`, which has 8,466).
- Applied the same ten deterministic feature-coordinate masks to every model and averaged scaled cosine reconstruction error per node.
- Measured pairwise Spearman correlation of per-node errors.
- For an interpretable set-overlap view, marked each model's highest-error 20% of nodes and measured pairwise Jaccard overlap. Under independent 20% selections, expected Jaccard is 0.111.

## Main findings

The different source-trained FP models overwhelmingly find the same target nodes difficult.

- Mean pairwise per-node error-rank correlation is **0.911** across targets (range **0.868–0.938**).
- Mean Jaccard overlap between hardest-20% node sets is **0.787** (range **0.621–0.988**), versus **0.111** under independent selections.
- A pair of models identifies the same node as hard **4.36×** as often as independence predicts.
- **12.8%** of nodes are in the hardest quintile for all nine source models. Because each individual model contributes 20%, this is strong consensus.
- **20.1%** are hard for a majority of models, while **73.1%** are hard for none.

Agreement is nearly complete on `covid19_twitter` and `midterm`: mean hardest-set Jaccard is **0.988** and **0.987**. It is lowest, but still very high, on Facebook (**0.621**), COVID Political (**0.649**), and Election (**0.649**).

Allowing an oracle to select the lowest-error source model separately for every node reduces mean error only **2.5%** relative to the best single source model (target range **0.5–8.0%**). So even where source models differ, their node-level complementarity is small.

## Interpretation

This strengthens the earlier aggregate result that FP is predominantly target-dependent. The source graph can move the absolute error level, especially on Facebook, Election, and COVID Political, but it changes the ordering of easy and hard target nodes surprisingly little. In this node-only FP setting, target feature geometry appears to determine difficulty much more strongly than source pretraining provenance.

The contrast with LP should be stated carefully. The LP analysis used thresholded classification mistakes, whereas this FP analysis uses continuous reconstruction errors and a fixed hardest quintile, so the absolute overlap numbers are not directly commensurate. Qualitatively, however, FP shows much stronger cross-source consensus, consistent with LP being more sensitive to source–target interaction.

## Caveats

- These are retrained seed-0 specialists, because the earlier aggregate FP evaluator did not retain per-node outputs.
- Results cover node-only MLPs and the repaired Order C protocol; they do not establish the same behavior for neighborhood models or GraphSAGE.
- The 20% cutoff is a diagnostic convention. The rank-correlation result does not depend on that cutoff.
- The oracle is diagnostic, not a realizable selection rule.
