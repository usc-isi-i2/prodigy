# Findings: does graph similarity predict single-source NM transfer?

*Pilot result. Companion to [`README.md`](./README.md) (methodology/caveats) and the
[notebook](../../plotting/similarity_vs_transfer/similarity_vs_transfer.ipynb). Last updated 2026-07-09.*

![Similarity vs. transfer summary](../../plotting/similarity_vs_transfer/figures/similarity_transfer_slide.png)

## Takeaway

**How different two retweet graphs are predicts how well a model pretrained on one
transfers to the other — and it is the *feature* axis, not raw topology, that does the
predicting.** More-divergent source ⇒ lower NM transfer, for every similarity metric.
The strongest, most robust predictor is **feature-cloud separability
(proxy-A-distance, mean ρ ≈ −0.92)**; raw **degree-distribution (topology) distance is
the weakest** (ρ ≈ −0.6) and even flips sign for one target. This is consistent with
our earlier feature-ablation result that NM leans on feature *content* while topology
alone ≈ chance — two independent analyses now point the same way.

Treat this as **directional evidence, not a powered result**: N = 20 directed pairs
across 4 independent target columns, one task (NM), one budget. It justifies — does not
replace — the interventional single-axis sweep.

## Results

Primary analysis = **within-target Spearman ρ**: fix a target, rank its candidate
sources by transfer accuracy, correlate against similarity(source, target). Doing it
per-column differences out target difficulty (divergence is symmetric; transfer is
directed). Expected sign is **negative**. Mean ρ across the 5 target columns:

| Similarity axis | mean ρ (with self) | mean ρ (self excluded) | sign-consistent targets |
|---|---:|---:|---:|
| **proxy_a_distance** (feature separability) | **−0.92** | **−0.86** | 5 / 5 |
| **homophily_gap** (signed, directional coupling) | −0.52 | **−0.80** | 5 / 5 (self-excl.) |
| feat_frechet | −0.84 | −0.66 | 4 / 5 |
| feat_mmd2 | −0.84 | −0.66 | 4 / 5 |
| feat_centroid_cosdist | −0.72 | −0.48 | 4 / 5 |
| indegree_ks (topology) | −0.64 | −0.46 | 4 / 5 |
| outdegree_ks (topology) | −0.60 | −0.42 | 4 / 5 |

Reading the numbers:

- **`proxy_a_distance` is the standout.** It barely moves when the trivial in-domain
  (self) cells are removed (−0.92 → −0.86), so it tracks real cross-graph structure,
  not just "in-domain wins."
- **`homophily_gap` — the one directional/signed predictor — *strengthens* when self
  is dropped** (−0.52 → −0.80). That is the pattern expected from a coupling mechanism:
  it has nothing to say about the trivial self case (gap ≈ 0) and only shows its hand on
  genuine cross-graph pairs. First empirical hint that an asymmetric predictor carries
  information a symmetric distance cannot.
- **Topology is weakest and least robust**, and **`cp_hk` is the lone anomaly**
  (ρ = +0.2 / +0.4): sources topologically *closer* to `cp_hk` transfer *worse* to it.
  `cp_hk` is the small, high-reciprocity, assortativity-outlier graph — flag before
  trusting topology-only predictors near it.

Underlying transfer signal (NM, 30-way, 3-shot, test; **accuracy** used as DV because
ROC-AUC is near ceiling): self-transfer is best in every column (covid→covid 0.66,
ukr→ukr 0.52, twibot20→twibot20 0.48, midterm→midterm 0.42); `cp_hk` is the hardest
target for all sources (0.12–0.16). Full matrix:
[`transfer_matrix.csv`](../../plotting/similarity_vs_transfer/transfer_matrix.csv),
joined with similarity in
[`joined_transfer_similarity.csv`](../../plotting/similarity_vs_transfer/figures/joined_transfer_similarity.csv).

## How we got the data / what it represents

Two stages, both on Tucker (see [`graph_divergence/README.md`](../graph_divergence/README.md)
and this experiment's [`README.md`](./README.md)):

- **Similarity** — [`compute_graph_divergence.py`](../graph_divergence/compute_graph_divergence.py)
  memory-maps each retweet graph (up to ~75 GB), subsamples nodes/edges (seed 0;
  feat_sample=4000, edge_sample=300k, mmd_cap=1000), and computes pairwise divergence on
  three axes: **topology** (in/out degree-distribution KS), **features** (GTE bio-embedding
  cloud distance: centroid cosine, Fréchet, RBF-MMD², and proxy-A-distance = a logistic
  domain classifier's separability of the two clouds, 0 = indistinguishable → 2 = perfectly
  separable), and **feature–structure coupling** (edge feature-homophily vs. random
  baseline). Generated 2026-07-03 → `graph_divergence_data.json`.
- **Transfer** — 4 single-source NM checkpoints (covid, ukr, midterm, twibot20) evaluated
  on 5 targets at one matched regime (30-way / 3-shot / test), pulled from existing
  `metrics_test_step0.json` eval runs.
- **Join** — done **within-target (column-wise)** on purpose (see Takeaway); the primary
  DV is NM top-1 accuracy, which spans 0.12–0.66 and is far more discriminative than the
  near-ceiling ROC-AUC.

Regenerate the figure:
`~/.pyenv/versions/myenv/bin/python scripts/plotting/similarity_vs_transfer/make_slide_figure.py`.

## Next steps

The pilot's purpose was to decide whether the signal is worth GPU time before committing
— it is. Roughly in order:

1. **Interventional single-axis sweep** (the causal test) — hold everything fixed and
   vary topology vs. feature content to show feature divergence *causes* the transfer gap
   rather than merely correlating. This is what the `experiment/topology-feature-ssl`
   branch is set up for.
2. **Kill the confound (cheapest hardening):** train matched-budget **midterm-only /
   cp_hk-only** (± political-only) single-source models so every matrix cell comes from
   the same run family, then re-check ρ on independent data.
3. **Generalize beyond NM / one budget** — repeat on LP/PL or a second budget.

## Caveats (flag these when presenting)

- **Small N.** 20 directed pairs, 4 independent columns. Descriptive, not a powered
  hypothesis test — do not over-read any single ρ.
- **Training-family confound.** `midterm` and `twibot20` *source* checkpoints come from a
  different run family (`nm_cm_midterm`, `nm_twibot20`) than `covid`/`ukr` (`nm_matrix_*`);
  budget/steps not guaranteed matched. The within-target design mitigates it (target
  difficulty differenced out; proxy-A-distance stays 5/5 sign-consistent) but does not
  fully rule it out. This is the main thing a critical reviewer will hit → motivates
  next-step 1 or 2.
- **One task, one budget.** NM only; the LP/PL merged-vs-single story is separate.
- **`cp_hk` topology sign-flip** is an unexplained anomaly, not yet trusted.
