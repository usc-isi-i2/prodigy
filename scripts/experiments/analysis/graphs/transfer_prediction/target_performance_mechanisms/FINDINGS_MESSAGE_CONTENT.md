# Background-message presence matters more than the tested neighbor content

7 September 2026. Completed frozen-model controls, not new training.

## Contribution-level finding

The large support-deletion effect is not reproduced by degree-preserving
rewiring, by replacing neighbor-specific messages with a subgraph-wide mean,
or by switching the observed sum operator to actual mean aggregation. Even
bias-only messages retain performance much closer to intact inference than
to deletion. This separates **retaining a background-message pathway** from
**the particular connections and neighbor content tested here**.

The sign remains task-dependent: deleting this pathway repairs Hong Kong
political and page classification, but hurts bot-detection ranking. This is a
useful diagnosis of the evaluated checkpoints, not proof that topology or
features are universally irrelevant, or that one training-source property
caused the learned behavior.

## Complete support-control results

AUC-point changes from intact HK inference; ranges over three initialization
seeds and two cached streams. Queries and all weights remain fixed.

| Support messages | covid-political | facebook-page-reference | twibot20 |
|---|---:|---:|---:|
| Actual mean aggregation | -0.270 to +0.040 | -0.004 to +0.170 | -0.529 to -0.218 |
| Remove affine message bias | -5.324 to +1.022 | -2.297 to +0.260 | -12.361 to -1.151 |
| Bias only | -1.110 to -0.362 | -0.245 to +0.138 | +0.130 to +0.811 |
| Subgraph-mean message | -0.455 to -0.032 | -0.038 to +0.067 | -0.220 to -0.067 |
| Zero messages | +4.651 to +11.101 | +0.885 to +2.035 | -11.296 to -6.550 |

Zero messages reproduce the corresponding edge-removal metrics for query-only,
support-only and joint interventions. An affine message is `W x + b`: the
no-bias control retains `W x`, bias-only retains `b`, and subgraph-mean assigns
all senders the mean projected feature of the real nodes in their own sampled
subgraph. Node-specific self projections, sampled members, pooling and the
downstream computations remain unchanged.

For Ukraine, support-only actual mean changes political AUC by +0.002 to
+0.046 points and page AUC by -0.058 to +0.041 points. On bot detection it
loses 1.392-1.988 points; zero messages lose 4.723-6.265 points. All five
targets, both sources, all roles and all signs are retained in the 960-cell
export and 160-row source summary, not selected after seeing this table.

### What this rules out

- The observed mean/sum mismatch is real, but switching to actual mean does
  **not** recover the political or page deletion gain. Do not attribute the
  headline result to that implementation mismatch alone.
- The repair is not a simple affine-bias-removal recipe. Removing only bias
  gives mixed political/page AUC effects, although HK political NLL improves
  in every seed/stream. Probability loss and ranking are distinct outcomes.
- Bias-only messages remain edge/degree-dependent and preserve the learned
  message branch. They are not equivalent to no graph information: sampled
  membership, self features and pooling remain available.
- Subgraph-mean replacement still contains sampled feature information. Tiny
  AUC changes do not prove logits, every prediction, or individual inputs are
  invariant. The claim is about relative target-performance sensitivity.

## Input-degree and label audit

Both checkpoints process exactly the same cached target inputs. Thus an
HK-versus-Ukraine difference in *target* degree or homophily cannot explain
their different responses; it is their learned processing that differs.
These descriptors do not replace a comparison of the original training corpora.

| Target | Real support-node occurrences with in-degree > 1 | Edges with both labels | Same-label fraction among labeled edges |
|---|---:|---:|---:|
| covid-political | 3.67-3.78% | 100% | 90.24-90.47% |
| election2020-political | 3.83-3.93% | 100% | 95.17-95.17% |
| facebook-page-reference | 10.98-11.16% | 64.22-65.31% | 17.74-20.72% |
| twibot20 | 9.06-9.13% | 4.29-4.31% | 44.83-45.03% |
| ukraine-suspended | 11.10-11.20% | 84.89-85.12% | 55.28-55.63% |

These are occurrence-weighted sampled-subgraph descriptors, not full-network
statistics or independent samples. Pooling nodes are excluded from degree
counts. Agreement uses raw graph labels and is not adjusted for class balance;
FB remains multiclass in this descriptor despite binary episode-local tasks.
T20's sparse label coverage prevents treating its agreement as representative
of all neighbors. Target labels are used only after evaluation for descriptors,
never to select interventions or construct predictions. All 320 batch hashes
and all center-label alignments were checked. Most political nodes have zero
or one incoming edge, which limits how broadly actual mean versus sum changes
node-level aggregates; it does not by itself explain their downstream effects.

## Completion and provenance

- Full message panel: 960 cells, six source/seed checkpoints, five targets,
  two streams, sixteen conditions. No checkpoint was trained or selected.
- Every one of 60 model/target/stream baselines and its three zero-message
  controls matches the independently saved role-topology metrics to 1e-6.
- Ten role-local direct-forward checks per cell validate the factored replay;
  untouched-role encodings, weights, buffers and operators are restored.
- Each runtime aggregation audit confirms `SumAggregation`, despite the
  configured `aggr="mean"`, under PyG 2.3.1. The checkpoint does not establish
  the training-time library version. Production model code was not patched.
- Message runtime revision `ca3fab25`; completed in 595.976 seconds. Analysis
  and input audit revision `accc241f`. All CPU-only; no GPU jobs launched.
- Compact results: `data/message_content_20260907/` and
  `data/role_input_audit_20260907/`. Per-query logits and receipts remain in
  `/dataMeR1/phil/gfm/prodigy-role-topology/log/message_content_full_20260907`.
- Branch `codex/role-topology-interactions`; local worktree
  `/Users/philipp/projects/gfm/prodigy/.worktrees/role-topology`; Tucker
  worktree `/dataMeR1/phil/gfm/prodigy-role-topology`.

## Publication consequence and remaining discriminating work

Lead with replicated role-specific intervention benefits and the new
presence-versus-content distinction. Treat exact label reconstruction as an
implementation check, not a surprising discovery. The second nonpolitical
target weakens a politics-only account without resolving all feature/label
construction concerns. Demote the readout selector to an appendix comparison.

The shared edge-retention dose curve and saved-checkpoint intervention study
are now complete: see [FINDINGS_SUPPORT_DOSE.md](FINDINGS_SUPPORT_DOSE.md). Source-training
degree/label descriptors, direct feature-label overlap checks and a portable
end-to-end release still require work. Do not keep rerunning the completed
rewiring or message grid, claim population significance from the two streams,
or imply that the aggregate-results companion reproduces model evaluation.
