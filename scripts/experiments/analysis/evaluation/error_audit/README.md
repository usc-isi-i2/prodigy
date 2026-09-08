# Error-audit analysis

`build_error_report.py` joins exported graph-node ids to the graph's stable user ids
and to the exact bio-selection policy used at graph construction.  It writes:

- `enriched_predictions.jsonl`: every prediction, including raw bios;
- `report.html`: balanced diagnostic cards;
- `summary.json`: record/group/profile coverage counts.

## Findings

- [Canonical-split NM audit](FINDINGS_NM_CANONICAL_SPLIT.md): corrected held-out
  edges, identical realized model inputs, paired failures, bio groups, query
  weighting, and GTE distributions. This supersedes the original NM diagnostics.
- [Historical NM bio clusters and query weighting](FINDINGS_NM_BIO_CLUSTERS.md): semantic
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
- [Historical Ukraine versus Hong Kong neighbor matching](FINDINGS_NM_SOURCE_PAIR.md):
  exact paired 30-way query/support ids, native-source specialization, shared
  error overlap, and cross-model oracle headroom on both source graphs.

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
