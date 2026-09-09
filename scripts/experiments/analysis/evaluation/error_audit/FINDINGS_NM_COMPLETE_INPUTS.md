# Canonical NM: complete sampled-input failure analysis

8 September 2026. Reconstructed all 512 held-out test episodes per target from
the original plans and RNG resets: 215,040 sampled subgraphs and 122,880 query
occurrences across Ukraine and Hong Kong. Both complete input hashes match the
original canonical audit exactly. Every saved batch was reloaded and verified
against its original tensors. No model forward passes or training were run.

## Main result

**Relative similarity of complete sampled neighborhoods separates correct and
incorrect predictions much more clearly than the earlier center-text comparison.**
This is an input-level association, not proof of a causal failure mechanism or
evidence that a mean-feature classifier improves the model.

For each sampled query/support subgraph, average all real nodes' original
768-dimensional feature vectors, excluding the synthetic pooling node. Normalize
that mean. Score each of the 30 candidate classes by the mean cosine to its
three support summaries. The margin is the true-class score minus the largest
of the 29 competing scores. No trained model or outcome enters these scores;
the ground-truth class is used only to categorize the margin afterward.

| Target | Input geometry | Query occurrences | Native-model accuracy |
|---|---|---:|---:|
| Ukraine | True class strictly closest | 13,948 | **82.35%** |
| Ukraine | Rival class strictly closer | 28,463 | **25.39%** |
| Hong Kong | True class strictly closest | 2,525 | **37.70%** |
| Hong Kong | Rival class strictly closer | 27,183 | **15.22%** |
| Hong Kong | Tied within tolerance | 2 | 0.00% |

These comparisons require a nonzero query summary and nonzero summaries for all
90 supports. Coverage is **42,411/61,440 (69.03%)** on Ukraine and
**29,710/61,440 (48.36%)** on HK. Other queries are excluded, not counted wrong.
Strict comparisons use a cosine tolerance of 1e-7. Among errors on these eligible
rows, a rival has a higher score in 89.61% Ukraine / 93.60% HK. This is a prevalence
within the selected rows, not a fraction of all errors causally explained.

Absolute cosine is usually high; margins against competitors are more informative
than a query's absolute similarity to its own supports. The summary is invariant
to node ordering and does not represent the full topology. It is neither the
encoder's learned embedding nor a literal concatenation of the full model input.

## Checks beyond degree and repeated-query ambiguity

The native accuracy gap holds within every populated degree bucket:

| Full-graph incident degree | Ukraine: true closest / rival closer | HK: true closest / rival closer |
|---|---:|---:|
| 1–9 | 82.37% / 37.16% | 66.43% / 42.64% |
| 10–99 | 87.96% / 37.10% | 42.32% / 19.29% |
| 100–999 | 86.92% / 29.44% | 35.20% / 13.66% |
| 1,000–9,999 | 81.52% / 23.64% | 29.00% / 12.43% |
| 10,000–99,999 | 66.22% / 18.47% | 29.87% / 13.22% |
| 100,000+ | 46.86% / 14.10% | — |

Bucket counts are retained in the aggregate JSON. These are descriptive strata,
not covariate-balanced experiments. Among queries seen both correct and wrong
within the eligible rows, correct occurrences have larger average margins:
mean paired difference +.00998 for 2,306 Ukraine queries and +.00577 for 1,033 HK
queries (each query equally weighted). Query identity fixes degree, but anchors,
supports, contexts and candidate classes still vary.

Restricting to queries assigned only one anchor within their episode retains
the pattern: Ukraine 82.61% (13,827 rows) versus 25.78% (27,508); HK 40.81%
(2,144) versus 16.87% (20,274), plus two tied rows. This excludes observed
multi-anchor query-role occurrences, not every possible neighborhood overlap.

## Exact node overlap gives independent descriptive evidence

Compute Jaccard similarity between a query's sampled node-ID set and the union
of a candidate class's three sampled support node-ID sets. Centers are included;
synthetic pooling nodes are excluded. Every query has a valid comparison here.

| Target | Overlap geometry | Occurrences | Native accuracy |
|---|---|---:|---:|
| Ukraine | True class strictly largest Jaccard | 16,382 | 77.21% |
| Ukraine | Rival strictly larger | 42,406 | 31.26% |
| Ukraine | Tied | 2,652 | 46.00% |
| Hong Kong | True class strictly largest Jaccard | 3,655 | 34.06% |
| Hong Kong | Rival strictly larger | 54,976 | 15.28% |
| Hong Kong | Tied | 2,809 | 47.35% |

The query center itself appears somewhere in a rival support's sampled subgraph
for **38.99% of wrong versus 16.09% of correct Ukraine predictions**, and
**78.99% versus 60.65% on HK**. This includes rival support centers as well as
their context nodes, and is associated with query degree. It does not establish
label leakage, a conflicting support label, or a causal distracting-node effect.
The sampled background edges remain the canonical static-training edges.

Whole-input sizes are not radically different on Ukraine: mean real query-node
counts are 82.89 wrong / 81.34 correct. HK wrong queries have larger sampled
subgraphs on average, 76.50 / 66.17. Mean support-union intersection counts alone
do not have a common direction across targets; relative overlap against rivals
is the more consistent diagnostic.

## What remains unexplained

- HK still gets 62.30% of the favorable mean-feature cases wrong. Favorable raw
  summary geometry is not sufficient for the learned decision to be correct.
- On Ukraine inputs with the true class closest, the Ukraine model gets 82.35%
  correct versus 33.08% for the Hong Kong model. On HK favorable inputs, the
  corresponding figures are 37.03% for Ukraine and 37.70% for HK. Models can use
  the same input evidence differently, but this does not identify how training
  sources caused that difference.
- Neighborhood averaging removes edges and feature multimodality. We have saved
  the full edges and mappings for subsequent encoder/metagraph analysis, but
  this run has not traced learned representations or intervened on input features.
- Context-only means and concatenated normalized center/context means are in the
  aggregate results. Their valid populations differ, so their conditional
  accuracies should not be compared as if they were matched model ablations.

The [completed fixed-query support-geometry follow-up](FINDINGS_NM_SUPPORT_GEOMETRY.md)
tests whether changing supports changes this margin and the model outcome
together. It finds a clearer relationship on Ukraine than HK; the cross-episode
association here is not a universal explanation of the intervention results.
Paired-support training remains a proposed intervention, not a demonstrated remedy.

## Artifacts, provenance and validation

[Aggregate results and receipts](data/canonical_split/nm_complete_input_geometry.json)
· [aggregation script](summarize_nm_complete_inputs.py)
· [runtime setup](../../../setup/nm_complete_input_audit/README.md).

Private Tucker output:
`/dataMeR1/phil/gfm/error_audit/nm_complete_inputs_20260908_v2/`.
There are 32 compact batch files totaling 1,155,180,608 bytes. They preserve all
non-feature input tensors, complete global sampled IDs, edges, PyG boundaries,
center/pooling positions, labels and role mappings. Feature values are recovered
by indexing the exact backing merged graph, with −1 pooling IDs replaced by
zeros. This is a complete recoverable input archive tied to that feature
artifact, not a standalone feature snapshot. Recheck the recorded full-input
hash if the backing artifact is ever replaced.

For HK alone, the 512 episodes contain 107,520 sampled subgraphs: 30 classes x
(3 supports + 4 queries) per episode. The 16 compact files store 8,037,529 node-ID
integers, of which 7,930,009 are real sampled-node occurrences and 107,520 are
the synthetic pooling-node sentinel. They store 8,477,571 directed background
edges plus 215,040 directed node/pooling links. These are occurrence counts;
the same graph node can appear in many subgraphs, so they are not counts of
distinct HK nodes. Original anchor IDs are retained in `task_label_map` and are
independently checked against the private query/anchor table. This is why saving
the actual sampled input costs much more than recording the 30 anchor centers,
while remaining far smaller than duplicating all 768-dimensional feature rows.

The receipt records expected, reconstructed and reloaded full-input hashes,
per-file hashes, feature path/shape and input-table hashes. Independent joins
verify all query/anchor identities and both models' original outcomes. Recomputed
center-to-support cosines reproduce the earlier independently extracted values
within 6.0e-7 / 6.6e-7 on 49,243 / 45,225 comparable rows. All 61,440 rows per
target and model accuracies reconcile with the original canonical audit.

Successful runtime: `19a9a6a8`, 544.18 seconds after dataset/trainer initialization,
CPU with four threads. The shared TrainerFS initializer loads the configured
Ukraine checkpoint on CPU; no forward pass uses it. The first attempt was
interrupted during graph loading to preserve tensor-field order in the compact
format; it is excluded and retained separately. No thresholds were tuned on
outcomes and no hash tolerances were relaxed (input hashes must match exactly).

Runtime branch `codex/nm-complete-input-audit-20260908`; local runtime worktree
`/private/tmp/prodigy-nm-support-resampling`; Tucker worktree
`/dataMeR1/phil/gfm/prodigy-nm-complete-input-audit-20260908`.
Code moved via private Git only. Findings and helper copies are in
`/Users/philipp/projects/gfm/prodigy`, branch `main`. No private node-level inputs
are added to git. One checkpoint seed per model and exploratory comparisons;
no confidence intervals or causal training-source claims are made.
