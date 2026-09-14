# Input distribution and connectivity audit — 2026-09-12

The eight graphs have measurably different input distributions. Neighborhood
features make graph identity easier to distinguish, but shift magnitude alone
does not explain which pairs benefit from joint training.

## Scope and execution

- Eight exact seed-zero nonzero graph caches used by the MLP; Election excluded.
- 20,000 distinct uniform nodes per graph, raw embedding and exact cached
  node-plus-context-neighbor-mean input.
- 10,000 sampled supervision edges and 50,000 exact uniform nonedges per graph.
  Positive edges sampled with replacement; all known edges excluded from negatives.
- No new MLP training, checkpoint selection, or validation/test evaluation.
  Historical transfer results are overlaid separately after the input audit.
- Tucker CPU run completed in 317.9 seconds at code revision `ea8fe8c`.
- Remote worktree: `/dataMeR1/phil/gfm/mixture-scaling-input-shift` (detached).
- Remote output: `/dataMeR1/phil/gfm/mixture-scaling/results/input_shift_audit_20260912`.
- Local branch/worktree: `codex/mlp-input-shift-audit`, `/tmp/mlp-pair-error-code`.
- Full protocol: [input_shift_audit.md](../../docs/input_shift_audit.md).

## How much does p(x) vary?

Raw-node normalized sliced Wasserstein distances range from **0.0528 to 0.3040**,
or **2.90–15.78 times** the matched-size within-graph sampling reference. For
node-plus-neighbor input, distances range from **0.0613 to 0.3967**, or
**3.12–22.65 times** the reference. These ratios are descriptive, not significance
tests. Each view uses its own pooled projection scale.

Ukraine–COVID is the closest pair in all four views, including the sampled pair
summaries. Political–Facebook is the most separated raw-node pair; TwiBot20–Facebook
is the most separated node-plus-neighbor pair.

Held-out linear graph-classifier AUC (0.5 is chance):

| Graph pair | Raw node | Node + mean neighbors |
|---|---:|---:|
| Ukraine–COVID | 0.5932 | 0.6997 |
| Ukraine–Facebook | 0.9402 | 0.9599 |
| Political–Suspended | 0.8807 | 0.9536 |

Adding neighbors increases AUC for **28/28 pairs**, with a median increase of
0.0829. Within-graph pseudo-domain controls span **0.4844–0.5160**, near chance.
This establishes detectable distribution differences under these diagnostics,
not lack of support overlap or task-relevant incompatibility.

There is an obvious structural component: sampled nodes with **zero context
neighbor mean** range from **15.745% in TwiBot20 to 61.085% in Facebook**. The
original node embeddings are nonzero. These are isolates in the fixed context
view, not necessarily isolates in the full graph. Consequently, the additional
graph discrimination cannot be attributed solely to differences in neighbor
semantics. Matching context availability and separating neighbor marginals from
center–neighbor dependence would be a useful subsequent control.

## Do similar nodes connect?

Yes, relative to each graph's own uniform-nonedge background, in all eight graphs.
Using raw endpoint cosine as an edge score gives the following descriptive AUCs
on sampled supervision edges/nonedges (these are not held-out model evaluations):

| Graph | Cosine edge-score AUC | Mean edge cosine | Mean nonedge cosine |
|---|---:|---:|---:|
| Ukraine | 0.6791 | 0.5421 | 0.4961 |
| COVID | 0.7096 | 0.5507 | 0.4956 |
| Midterm | 0.6707 | 0.5785 | 0.5300 |
| Political | 0.6299 | 0.6017 | 0.5659 |
| Suspended | 0.6427 | 0.5476 | 0.5071 |
| TwiBot20 | 0.7323 | 0.5782 | 0.5119 |
| Hong Kong | 0.6768 | 0.5542 | 0.4995 |
| Facebook | 0.7680 | 0.5654 | 0.4755 |

Facebook shows the strongest edge/nonedge cosine separation despite Political
having the highest absolute edge cosine. This is why a graph-specific background
matters. Shared embedding clusters also have more same-cluster edges than the
product-of-endpoint-marginals expectation in every graph. Cluster mixing is a
coarse descriptive summary, with sparse cells masked; it is not a test of equal
p(y|x) across graphs.

## Does greater shift imply more negative transfer?

Not in these historical results. Interleaving beats the better constituent's
six-other-graph mean in **19/28 pairs**. Descriptive Spearman correlations between
input distance and transfer delta are **+0.296** for raw nodes and **+0.402** for
node-plus-neighbor inputs. Pairs share sources and omit different targets, so we
do not attach independence-based p-values or claim a causal relationship.

Two contrasting examples have nearly identical distances in the task-pair
node-plus-neighbor summary:

| Pair | Pair-input distance | Transfer delta vs best constituent |
|---|---:|---:|
| Political–Suspended | 0.2469 | +3.3411 pp |
| Ukraine–Facebook | 0.2453 | −1.6121 pp |

Magnitude of input shift alone therefore does not distinguish these two outcomes.
This does **not** establish conditional shift: finite model capacity, optimization,
training exposure and the particular direction of input shift remain possible
contributors. The pair representation is a symmetric diagnostic summary, not the
MLP's exact pair function, and loses some endpoint alignment information.

## Figures and evidence

- [Input distance matrices](figures/input_distances.png)
- [Graph-classification AUC matrices](figures/graph_discrimination.png)
- [Context availability](figures/context_availability.png)
- [Edge versus nonedge cosine distributions](figures/edge_similarity.png)
- [Shared-cluster edge mixing](figures/cluster_mixing.png)
- [Shift versus historical transfer](figures/shift_vs_transfer.png)
- [Full compact audit and receipts](data/audit.json)
- [Completion record](data/COMPLETE.json)
- [Historical comparison table](data/shift_vs_historical_transfer.csv)

Validation: graph identities/context receipts matched; finite feature and sample
size checks passed. Focused checks passed for exact nonedge rejection, symmetric
pair projection, zero identical-input distance, chance identical-domain AUC and
near-perfect AUC on a known synthetic shift. All six figures were rendered and
visually inspected. Raw node IDs, sampled features/pairs and projections remain
on Tucker; compact historical input tables and original file hashes are archived
under `data/` for reproducibility.
