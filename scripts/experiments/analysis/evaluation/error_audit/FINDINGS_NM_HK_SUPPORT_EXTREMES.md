# HK: deliberately selecting support similarity extremes

Nearest learned-space supports improve replacement quality, but unconditional
replacement remains harmful to the selected balanced sample. Similarity-guided
selection is not yet a sufficient fix for original failures.

## Native-model outcomes

All 200 original cases are reused: 100 incorrect queries and 100 correct controls.
Nearest/farthest/diverse use one triple per query; random uses five triples per
query, with every draw included in its rate.

| Selection | Rescued originally incorrect | Broke originally correct | Accuracy on balanced cases |
|---|---:|---:|---:|
| Nearest | 25/100 (25%) | 54/100 (54%) | 71/200 (35.5%) |
| Farthest | 2/100 (2%) | 90/100 (90%) | 12/200 (6%) |
| Diverse | 15/100 (15%) | 75/100 (75%) | 40/200 (20%) |
| Random, all draws | 62/500 (12.4%) | 330/500 (66%) | 232/1,000 (23.2%) |

Original balanced accuracy is 50%. Nearest therefore improves over random
replacement by 12.3 percentage points but remains 14.5 points below keeping the
original supports. These are selected-case accuracies, not full NM benchmark
estimates. Nearest rescues 12.6 points more failures and breaks 12 points fewer
correct controls than the mean random draw.

Among the **71 queries never rescued in either previous resampling condition**,
nearest rescues **5/71 (7.0%)**, diverse 2/71, and farthest 0/71. Random rescues
9/355 draws (2.5%), reaching 8/71 queries at least once across five attempts.
Across all 100 failed cases, random reaches 28 at least once across five attempts.
The five-attempt union must not be compared as a single-draw rescue rate.

Mean native NLL over balanced cases: nearest 3.272, diverse 3.977,
random 4.164, farthest 5.637. Mean selected learned cosine respectively:
0.780, 0.646, 0.670, 0.548.

## Interpretation and limits

This intervention strengthens the evidence that learned query/support
compatibility affects predictions: deliberately selecting nearby supports helps
relative to distant or random alternatives while query and competing support
inputs remain fixed. It does not isolate cosine from other attributes of the
selected support neighborhoods, establish a complete explanation for failures,
or establish that prototype or readout replacement is the remedy.

The hard residual mostly survives: 66/71 previously persistent failures remain
incorrect under nearest selection. A useful next analysis can reuse these cached
logits to distinguish nearest supports that still fail to separate the true class
in the simple cosine head from cases where that head succeeds but native
metagraph inference fails. Avoid another broad graph-loading run for that check.

Selection uses the known true class, excludes all episode queries, the anchor and
original supports, and searches a uniformly capped bank of at most 32 alternatives
per case. “Nearest” means within this bank, not all neighbors. Candidate contexts
are fixed per identity and sampled from canonical static_train edges. No training
or independent model/target replication was performed.

## Reproduction and validation

Runtime revision `79931273`, branch `codex/nm-complete-input-audit-20260908`,
local runtime worktree `/private/tmp/prodigy-nm-support-resampling`, Tucker runtime
worktree `/dataMeR1/phil/gfm/prodigy-nm-complete-input-audit-20260908`. Analysis and
helper copies are in `/Users/philipp/projects/gfm/prodigy` on `main`.

Compute completed in 40.23 seconds on owned GPU 0, encoding 1,963 unique candidate
centers and evaluating 1,600 interventions. Canonical HK edges extracted without
loading merged features: train 829,082; validation 177,494; test 177,803. Their
union matches standalone full edges, and undirected split pairs are disjoint.
All 200 eligible pool sizes match the prior run and contain its alternative IDs.
All 200 baselines replay within the fixed probability tolerance; model state
remains unchanged. Nearest/farthest cosine bounds contain every evaluated triple.

Private output: `/dataMeR1/phil/gfm/error_audit/nm_hk_support_extremes_20260908/`.
The candidate bank archive preserves graphs, embeddings, selections and all
three heads' logits. The receipt records hashes and parity results.
Aggregate evidence: [nm_hk_support_extremes.json](data/canonical_split/nm_hk_support_extremes.json).
Recompute with `summarize_nm_hk_support_extremes.py --input-root <private-results>`.
