# Degree-preserving rewiring and a second-target support repair

7 September 2026. Complete frozen-model follow-up, not new training.

## Main results

Support-edge removal improves Hong Kong models on **two targets across all
three initialization seeds and both cached streams**. The second target is
facebook-page-reference, not another heuristic political-label dataset.
Preserving sampled nodes and every node's in/out degree while rewiring edges
does not reproduce those gains.

| Target | HK support removal: AUC-point change, range over 3 seeds x 2 streams | HK support rewiring: AUC-point change, range over 3 seeds x 2 streams x 3 draws |
|---|---:|---:|
| covid-political | +4.65 to +11.10 | -0.019 to +0.041 |
| election2020-political | -0.52 to +1.72 | -0.269 to +0.122 |
| facebook-page-reference | +0.89 to +2.03 | -0.008 to +0.044 |
| twibot20 | -11.30 to -6.55 | -0.079 to +0.084 |
| ukraine-suspended | -0.02 to +6.40 | -0.195 to +0.055 |

These are observed ranges, not confidence intervals. Rewiring draws and cached
episode streams are not independent training seeds. All values above are
support-only changes with query processing intact. Ukraine controls and all
mixed query/support conditions are retained in the exported tables.

On facebook-page-reference, Ukraine support removal also improves AUC in every
seed/stream, by +0.076 to +0.343 points, considerably less than Hong Kong.
Election and suspension do not show uniform HK support-removal benefits across
all six comparisons. Both sources lose bot-detection AUC under support removal.
The claim is a target-dependent repair, not a universally useful edge policy or
a new best page-classification system: the strong raw-feature baseline remains.

The conditional role interaction also needs qualification. The historical HK
checkpoint's query-only removal harmed political AUC in both streams; in the
new three-seed controls its political effect ranges from -7.37 to +4.71 points.
Likewise, removing both roles does not always beat support-only removal. The
opposite signs of isolated query/support removal **do** hold in all six HK page
comparisons: query removal loses 0.20-2.06 points while support removal gains
0.89-2.03. Do not promote the historical political sign pattern into a
three-seed universal claim.

## What the rewiring control rules out, and what it does not

Across support subgraphs, final edge turnover is 62.0-62.4% on covid-political,
78.8-79.0% on election, 39.5-40.3% on page classification, 76.2-76.5% on
twibot20 and 69.3-70.1% on suspension. In/out degree vectors are exact and
features, selected members and pooling connections are untouched. Fully
reciprocal subgraphs retain reciprocity. Rigid or empty subgraphs remain in
the panel and are counted explicitly; these are finite swaps, not uniform
random graph draws.

The repair is therefore not reproduced merely by replacing many particular
connections while preserving degree and sampled membership. This points away
from those connections alone as the explanation. It does **not** establish that
topology is irrelevant: degree, membership and some unswappable edges remain.
It also does not isolate degree, smoothing, feature homogeneity or affine
message terms as the cause of the deletion benefit.

## Representation observations

On covid-political, means across the fixed seed/stream episode panel:

| Quantity | HK intact / removed / rewired | Ukraine intact / removed / rewired |
|---|---|---|
| Unit-support within-class dispersion | .19572 / .11735 / .19573 | .13911 / .11693 / .13912 |
| Support class-mean cosine | .89668 / .93543 / .89668 | .81233 / .83474 / .81231 |
| Cosine to original support vectors | 1 / .89646 / .99992 | 1 / .97069 / 1.00000 |
| Learned label-pair cosine | .70705 / .88965 / .70710 | .52145 / .56041 / .52137 |

Rewiring barely moves these representations, whereas deletion changes HK much
more than Ukraine. The successful deletion increases label-pair cosine, so
"the repair makes class label vectors more separated" is not the explanation.
These are descriptive geometry checks, not a demonstrated mediation model.

## Verified implementation issue and next discriminating control

Read-only Tucker inspection found PyG 2.3.1 and the following concrete mismatch:
`SAGEConvSelfLoops(..., aggr="mean")` reports `aggr == "mean"`, but its actual
`aggr_module` is `SumAggregation`. Aggregating three unit messages with receiver
indices `[0, 1, 1]` returns `[1, 2, 0]`, not `[1, 1, 0]`.

`models/gnn_with_edge_attr.py` initializes `MessagePassing` without the requested
aggregation, then assigns `self.aggr`; the installed library's cached aggregation
module remains sum. `models/get_model.py` requests mean. This observation is
about the verified inference runtime; checkpoint tensors alone do not establish
the historical training library version. All replay baselines still exactly
match the existing evaluations at the metric tolerance.

No production code or checkpoint was patched. `message_content.py` adds scoped,
reversible **experimental controls**: actual mean aggregation, removing the
message projection's affine bias, bias-only messages, subgraph-mean messages,
and zero messages. Tests verify restoration and equivalence of zero messages
to edge removal. These controls have not yet produced target-performance
results. The immediate next experiment should distinguish degree-dependent
message scaling from neighbor-specific information before upgrading the causal
claim or retraining anything.

## Completion and provenance

- 1,140 cells: six existing HK/Ukraine controls, five targets, two streams,
  nineteen unique topology-role conditions. No model selection or training.
- All 60 model/target/stream baselines match independently saved AUC, accuracy,
  F1 and NLL to 1e-6; all input and model-weight checks pass.
- Direct role-specific rewiring forwards match factored predictions; untouched
  role embeddings are bit-exact. Edge-attribute-zeroing parity confirms the
  retained varying edge attributes are unused by this encoder.
- Tucker runtime revision: `9e93f59c`; elapsed 799 seconds; tmux process ended
  with `DONE.json`. Analysis revision: `68cf761b`.
- Compact tables and receipts: `data/role_topology_20260907/`. All 240
  conditional-effect rows, 300 geometry summaries and 20 topology summaries
  are retained. Raw per-query logits and per-subgraph receipts stay in
  `/dataMeR1/phil/gfm/prodigy-role-topology/log/role_topology_full_20260907/`.
- Branch `codex/role-topology-interactions`, local worktree
  `/Users/philipp/projects/gfm/prodigy/.worktrees/role-topology`; Tucker worktree
  `/dataMeR1/phil/gfm/prodigy-role-topology`. The paper PDF is not yet revised
  with this follow-up. The earlier results are preserved, not overwritten.
