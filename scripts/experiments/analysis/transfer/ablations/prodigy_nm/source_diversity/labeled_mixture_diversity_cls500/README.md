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
The five `mixture_diversity_trajectory_contains_<graph>.png` figures show every 1,000-step
donor subset containing the named graph, without averaging. The filter graph is absent
as an evaluation panel because a held-out target cannot also occur in its training mixture.
`mixture_diversity_trajectory_all_points.png` retains all 75 exact 1,000-step held-out
mixtures, while `mixture_diversity_trajectory_averaged_1000.png` averages those cells by
mixture size in macro and per-target panels.
`marginal_donor_effect_boxplot_1000.png` shows the 28 matched ROC-AUC changes from adding
each graph to otherwise identical nonempty donor subsets across the four other targets.
`marginal_donor_effect_heatmap_1000.png` gives the corresponding added-graph-by-target
matrix of mean matched effects, with seven subset additions averaged in each cell.
`marginal_vs_single_source_transfer_1000.png` compares that matrix with the same five
graphs from the historical 9×9 single-source transfer matrix; its caption records the
different training and evaluation protocols, so the comparison is pattern-level only.

```bash
/opt/homebrew/bin/python3.11 analyze.py \
  --heldout data/heldout_seed0_shard0.jsonl data/heldout_seed0_shard1.jsonl \
  --controls data/controls_seed0_shard0.jsonl data/controls_seed0_shard1.jsonl

/opt/homebrew/bin/python3.11 analyze_trajectory.py
```
