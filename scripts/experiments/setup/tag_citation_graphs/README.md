# Cora and PubMed smoke runs

These configs exercise the two GTE-attributed citation graph loaders after the
artifacts have been generated with
`scripts/graph_construction/generate_tag_citation_graph.py`.

Run a lightweight classification smoke on Tucker:

```bash
python -u experiments/run_single_experiment.py \
  --config scripts/experiments/setup/tag_citation_graphs/cora_cls_smoke.yaml

python -u experiments/run_single_experiment.py \
  --config scripts/experiments/setup/tag_citation_graphs/pubmed_cls_smoke.yaml
```

These are integration gates, not paper-result configurations. The graph
catalog keeps both datasets out of the default all-graph sweep until the built
artifacts and smoke runs are verified on Tucker.
