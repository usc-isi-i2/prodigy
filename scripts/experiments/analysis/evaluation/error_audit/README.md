# Error-audit analysis

`build_error_report.py` joins exported graph-node ids to the graph's stable user ids
and to the exact bio-selection policy used at graph construction.  It writes:

- `enriched_predictions.jsonl`: every prediction, including raw bios;
- `report.html`: balanced diagnostic cards;
- `summary.json`: record/group/profile coverage counts.

## Findings

- [HK NM mechanism and bounded repair](FINDINGS_NM_HK_GOAL.md): controlled
  key/value and direction/magnitude interventions, followed by a fixed geometry
  residual that improves selected modified cases but fails on original HK NM.
- [HK failure localization](FINDINGS_NM_HK_FAILURE_LOCALIZATION.md): complete-edge
  accounting of wrong anchors, nearest-support residuals, and cached true-class
  versus competitor score interventions.
- [Canonical NM source-by-stage comparison](FINDINGS_NM_SOURCE_STAGES.md): all
  original Ukraine/HK test inputs, both source checkpoints, common encoded heads
  versus final decisions, readout recoveries/losses and query-weighting sensitivity.
- [NM bio clusters and query weighting](FINDINGS_NM_BIO_CLUSTERS.md): semantic
  clusters, query/anchor/support GTE distributions, actual-text examples,
  the Hong Kong node-weighted ranking reversal, and overlapping anchor labels.
- [Query bio embedding clusters](FINDINGS_QUERY_BIO_CLUSTERS.md): outcome-blind
  GTE clustering with fresh-stream assignment, actual-text interpretation,
  error enrichment, and coarse label/isolation adjustment.
- [Consolidated source-paired error-audit report](SOURCE_PAIR_ERROR_AUDIT_REPORT.md):
  the complete cross-task synthesis, figures, caveats, and actionable next
  analyses.
- [Ukraine versus Hong Kong on COVID Political](FINDINGS_COVID_POLITICAL_SOURCE_PAIR.md):
  paired FP/FN cohorts across two fixed evaluation streams, with private
  node/profile evidence retained on Tucker.
- [Ukraine versus Hong Kong neighbor matching](FINDINGS_NM_SOURCE_PAIR.md):
  exact paired 30-way query/support ids, native-source specialization, shared
  error overlap, and cross-model oracle headroom on both source graphs.
- [Full HK class-reference decomposition](FINDINGS_NM_HK_CLASS_REFERENCE_FULL.md):
  all 512 canonical HK episodes, correct controls, ambiguity strata, and the
  native-versus-Ukraine positive-support decision pathway on identical inputs.
- [Native-graph class-reference replication](FINDINGS_NM_NATIVE_GRAPH_REFERENCE_REPLICATION.md):
  shows which HK findings replicate on Ukraine and separates pre-metagraph from
  final source divergence on identical target episodes.
- [HK training-conflict bridge](FINDINGS_NM_HK_TRAINING_CONFLICT_BRIDGE.md):
  connects canonical failures to verified consumed training episodes while
  separating established exposure, controlled interventions, and hypotheses.
- [NM failure hierarchy](FINDINGS_NM_FAILURE_HIERARCHY.md): integrates one-query,
  episode, full-HK, native/foreign, training-conflict, and nine-graph evidence
  while separating causal results from localization and graph-level correlation.
- [Matched HK overlap-aware training](FINDINGS_NM_HK_OVERLAP_TRAINING.md): exact
  baseline/treatment stream shows that deleting contradicted negative messages
  during training hurts terminal corrected and unique-anchor accuracy.
- [HK/Ukraine source-manifold affinity](FINDINGS_NM_SOURCE_MANIFOLD.md): five
  balanced reference-bank replicates reject global target-manifold proximity as
  a consistent explanation for native NM success.
- [Native/foreign margin stages](FINDINGS_NM_SOURCE_MARGIN_STAGES.md): paired
  candidate margins show that Ukraine's native advantage is mostly present
  before the metagraph, while HK's is mostly created by its readout.
- [Broad native/foreign stage audit](FINDINGS_NM_BROAD_SOURCE_STAGES.md): exact
  published randomized episodes extend stage localization to four native graphs;
  HK is the clearest readout-dominant case, while source matching also changes
  pre-metagraph ordering on every target.

For the current Parquet Twitter graphs:

```bash
conda activate bio-embeddings-v001  # provides DuckDB for the provenance join
python scripts/experiments/analysis/evaluation/error_audit/build_error_report.py \
  --predictions /dataMeR1/phil/gfm/error_audit/regression/midterm/midterm__reg_probe_examples.jsonl \
  --graph /dataMeR1/phil/data/midterm/graphs/retweet_graph_parquet.pt \
  --bio-root /dataMeR1/phil/data/midterm/bio_embeddings/gte-multilingual-base/version=v001 \
  --model MODEL_NAME --target followers_count \
  --out-dir /dataMeR1/phil/gfm/error_audit/reports/midterm_reg_followers
```

For an older classification graph backed by a CSV, use `--profile-csv`,
`--profile-id-column`, and `--profile-bio-column`.  COVID-political uses row-index
node ids and stores bios in `profile`, so use
`--profile-id-column __index__ --profile-bio-column profile`.

The HTML contains a half high-confidence/large-error and half deterministic-random
sample from each group.  The JSONL remains the complete evidence set.  Raw reports
stay cluster-local; only hand-redacted aggregate findings should be committed here.
