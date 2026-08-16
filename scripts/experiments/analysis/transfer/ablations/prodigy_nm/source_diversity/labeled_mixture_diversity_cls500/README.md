# Labeled-mixture diversity at fixed compute and along training

Analysis for `setup/labeled_mixture_diversity_cls500/`. The experiment trains every
nonempty proper subset of five labeled graphs for 500 optimizer steps and evaluates
only on absent targets using 500 paired 10-shot classification episodes. Models are
evaluated at 500, 750, and 1,000 total optimizer steps. Endpoint controls compare the
held-out four-source mixture with target-only and all-five pretraining.

Status: complete. See [`RESULTS.md`](RESULTS.md) for the interpretation.

`analyze.py` reproduces the original 500-step analysis. `analyze_trajectory.py`
validates 85 cells at each checkpoint, requires paired episode fingerprints across
all checkpoints, summarizes the diversity trajectories, and identifies models whose
750→1,000-step evaluation changes remain larger than 0.01 ROC-AUC.

Committed evidence includes the original four JSONL shards, continuation shards under
`data/trajectory/`, consolidated trajectory CSVs, and figures under `figures/`.
`mixture_diversity_trajectory_raw.png` shows every 1,000-step donor subset containing
TwiBot20 without averaging. TwiBot20 is absent as an evaluation panel because a held-out
target cannot also occur in its training mixture.

```bash
/opt/homebrew/bin/python3.11 analyze.py \
  --heldout data/heldout_seed0_shard0.jsonl data/heldout_seed0_shard1.jsonl \
  --controls data/controls_seed0_shard0.jsonl data/controls_seed0_shard1.jsonl

/opt/homebrew/bin/python3.11 analyze_trajectory.py
```
