# Similarity vs. transfer: does graph divergence predict single-source NM transfer?

Pilot analysis joining the graph-divergence results
([`scripts/experiments/graph_divergence/`](../graph_divergence/README.md)) with a
matched single-source neighbor-matching (NM) transfer matrix, to check whether
any similarity axis (topology, feature-marginal, or feature-structure coupling)
predicts which source graph transfers best to a given target — **before**
committing GPU time to filling out the full N×N single-source pretraining
matrix or running the interventional (causal) experiment.

## Why this needs care: symmetry vs. directionality

Our divergence metrics (KS distance, MMD, proxy-A-distance, Frechet, centroid
cosine) are **symmetric**: `d(A,B) = d(B,A)`. Transfer is **directed**:
`transfer(A→B) ≠ transfer(B→A)` in general. Pooling all directed pairs into one
scatter of `similarity` vs `transfer` is invalid: it places two different
y-values at the same x for every pair, so a symmetric x cannot explain the part
of the signal that is directional, and any fitted correlation is contaminated
by **target difficulty** riding along with the pair identity (e.g. a target that
is intrinsically hard to retrieve-30-way looks "far from every source" even
though the sources don't actually differ much among themselves).

**Fix: analyze within-target (column-wise).** Fix a target `t`, rank the
candidate sources by `transfer(s→t)`, and ask whether `similarity(s,t)` predicts
that ranking. This is the practically relevant question anyway ("given this
target, which source should I pretrain on?") and it differences out target
difficulty for free, because the target is held fixed within each column.

## Data

- **Transfer matrix**: [`scripts/plotting/similarity_vs_transfer/transfer_matrix.csv`](../../plotting/similarity_vs_transfer/transfer_matrix.csv).
  4 single-source NM models (`covid`, `ukr`, `midterm`, `twibot20`) × 5 targets
  (`covid`, `ukr`, `midterm`, `cp_hk`, `twibot20`), all at the **same eval
  regime** — 30-way, 3-shot, test split — pulled from
  `/dataMeR1/phil/gfm/prodigy/log/eval_<model>_to_<target>_nm_3shot_30way_*/data/metrics_test_step0.json`
  on Tucker. Includes the 4 in-domain (self) cells.
  - `midterm` and `twibot20` sources come from a different training-run family
    (`nm_cm_midterm`, `nm_twibot20`) than `covid`/`ukr` (`nm_matrix_*`) because no
    dedicated `midterm_only`/`twibot20_only` model was ever trained — only these
    matrix/cross-model runs exist. They were matched to the same eval regime so
    they're comparable, but budgets/steps are not guaranteed identical; treat as
    a pilot, not a controlled ablation.
  - NM `roc_auc` is close to ceiling for the easy pairs; the primary DV here is
    **NM accuracy** (30-way top-1), which spans 0.12–0.66 and is far more
    discriminative.
- **Similarity**: `scripts/plotting/graph_divergence/graph_divergence_data.json`
  (see that experiment's README) — pairwise `indegree_ks`, `outdegree_ks`,
  `feat_centroid_cosdist`, `feat_frechet`, `feat_mmd2`, `proxy_a_distance`, plus
  per-graph `feature_homophily` (used to build a signed coupling-gap predictor,
  `homophily_source − homophily_target`).

## Methodology

1. **Within-target Spearman.** For each target column (n=4 sources, including
   the self cell), correlate similarity(s, t) against transfer(s→t). Report the
   per-column rho for every metric, plus the sign-consistency and
   mean/median rho across the 5 columns. This is the primary analysis.
2. **Self-transfer robustness check.** Repeat with the in-domain (self) cell
   dropped from each column (n=3). The self cell has similarity ≈ 0 and usually
   the best transfer, so it can trivially anchor a negative correlation; if the
   correlation survives its removal, that's stronger evidence the signal isn't
   just "in-domain wins."
3. **Directional check (secondary).** Correlate the signed coupling gap
   (`homophily_source − homophily_target`) against transfer, since a symmetric
   distance cannot explain directional (asymmetric) transfer effects by
   construction — this is where an asymmetric predictor is expected to add
   information a symmetric one can't.
4. Everything reported **descriptively** (rho values, scatter plots) — with 4–5
   points per column this is not a powered hypothesis test. The purpose is to
   see if there's a signal worth chasing with a bigger matrix or the
   interventional experiment, not to publish a confirmed effect from N=20.

## Caveats

- N is small (20 pairs, 4 independent "columns" for the primary analysis) —
  do not over-read any single rho.
- `midterm`/`twibot20` source checkpoints are from a different run family than
  `covid`/`ukr` — a training-budget confound is possible and not fully ruled out.
- Only NM is compared here (the only task with a matched source/target grid);
  LP/PL findings from the merged-vs-single analysis are a separate story.

Notebook: [`scripts/plotting/similarity_vs_transfer/similarity_vs_transfer.ipynb`](../../plotting/similarity_vs_transfer/similarity_vs_transfer.ipynb).
