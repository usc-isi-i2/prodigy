# Labeled-mixture diversity at fixed compute

Analysis for `setup/labeled_mixture_diversity_cls500/`. The experiment trains every
nonempty proper subset of five labeled graphs for 500 optimizer steps and evaluates
only on absent targets using 500 paired 10-shot classification episodes. Endpoint
controls compare the held-out four-source mixture with target-only and all-five
pretraining.

Status: complete. See [`RESULTS.md`](RESULTS.md) for the interpretation.

`analyze.py` validates the complete 75-cell held-out matrix and 10 endpoint cells,
then writes standalone data, summaries, and figures beneath this folder.

Committed evidence includes the four raw JSONL shards, `all_results.csv`,
`summary.csv`, `endpoint_controls.csv`, `marginal_donor_effects.csv`, and
`figures/mixture_diversity_and_controls.png`.

```bash
/opt/homebrew/bin/python3.11 analyze.py \
  --heldout data/heldout_seed0_shard0.jsonl data/heldout_seed0_shard1.jsonl \
  --controls data/controls_seed0_shard0.jsonl data/controls_seed0_shard1.jsonl
```
