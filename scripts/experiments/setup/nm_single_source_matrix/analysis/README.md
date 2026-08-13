# Transfer-pairs analysis table

`transfer_pairs.csv` — one row per **directed** (source, target) pair of the 8×8 single-source
NM matrix (64 rows). Regenerate with `python3 build_transfer_pairs.py` (from repo root).

Sources: `../nm_single_source_matrix.csv` (ROC-AUC, rows=train/source, cols=test/target) joined with
`scripts/experiments/analysis/graphs/structure/graph_divergence/data/graph_divergence_data.json` (pairwise divergence + per-graph stats).
Transfer metric is **ROC-AUC**; the single-source pilot preferred top-1 accuracy (AUC near ceiling
in-domain), but off-diagonal AUC spans ~.55–.98 so regret is discriminative here. **1 seed** — treat
sub-~.03 gaps as noise.

## Columns

**Index**
- `source`, `target` — train graph → test graph. `is_self` = diagonal (in-domain) row.

**Core transfer**
- `transfer_auc` — AUC[source→target].
- `source_ceiling`, `target_ceiling` — in-domain AUC of each (diagonal). `target_ceiling` = **best model on target** (diagonal is the column-max in all 8 columns).

**Regret & retention** (difficulty-controlled against the target's own ceiling)
- `regret` = `target_ceiling − transfer_auc` (≥0; AUC points below best-on-target).
- `regret_norm` = `regret / target_ceiling` (= `1 − retention`).
- `retention` = `transfer_auc / target_ceiling` (fraction of achievable recovered).
- `donor_rank` — rank of this source among the 7 **foreign** sources into `target` (1=best); blank on self.

**Directional asymmetry** (this direction vs the reverse, source↔target swapped)
- `reverse_auc` = AUC[target→source].
- `auc_asym` = `transfer_auc − reverse_auc` (primary directional gap; + = this direction transfers better).
- `regret_reverse` = `source_ceiling − reverse_auc`; `regret_asym` = `regret − regret_reverse`.
- `retention_reverse`, `retention_asym` — same in retention units.

**Pairwise divergence / similarity** (symmetrized unless noted; 0 on diagonal)
- `proxy_a_distance` — directed as stored (domain-classifier separability, 0=identical→2=separable). **Primary similarity axis** (strongest transfer predictor).
- `proxy_a_distance_sym` — symmetrized (avg with transpose); use this one for analysis (raw is ~symmetric, ≤7% directional residual).
- `feat_frechet`, `feat_mmd2`, `feat_centroid_cosdist` — feature-cloud distances (symmetric).
- `indegree_ks`, `outdegree_ks` — degree-distribution KS (topology; weakest predictor).
- `homophily_gap_signed` = `feature_homophily[source] − feature_homophily[target]` — the one **signed/directional** metric; tracks the *sign* of `auc_asym` (ρ≈+0.66).

**Per-graph attributes broadcast onto the pair** (off-diagonal means)
- `source_centrality`, `target_centrality` — mean `proxy_a_distance_sym` to the other 7 (LOW = feature-central). The confound: donor rank ≈ centrality rank (ρ≈+0.95).
- `source_outflow` — source's mean AUC as donor (source strength; ∥ centrality, ρ≈+0.98).
- `target_inflow` — target's mean AUC as receiver (reachability; ⊥ centrality, ρ≈+0.21).
- `source_donor_regret_mean`, `target_recv_regret_mean` — the row-μ / col-μ margins of the regret table.
- `source_feat_homophily`, `target_feat_homophily` — underlie `homophily_gap_signed`.
- `source_label_homophily`, `target_label_homophily` — task easiness (blank where undefined: covid/ukr/midterm/cp_hk). elec/cvpol ≈ .95/.94 → near-trivial targets, easy to transfer *into* despite feature-distance.

## Key reads
- **Symmetry:** filter `is_self==False`; `auc_asym` (or `regret_asym`) is the directional gap; a pair and its swap mirror in sign.
- **Similarity control:** `source_centrality` vs `source_donor_regret_mean` is the donor-ranking confound; `source_outflow` vs `target_inflow` (decoupled) is why the asymmetry is *not* a similarity artifact.
