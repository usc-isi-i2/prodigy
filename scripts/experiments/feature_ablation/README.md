# Feature ablation: how much does each task rely on node features vs. topology?

**Question.** We pretrain mostly on Neighbor Matching (NM). Does NM (and do the
downstream tasks) actually *use* the node features (GTE bio embeddings), or are
they solved from graph topology alone? This gates whether adding a
topology-oriented pretraining task would add signal, and whether our features
are pulling weight at all.

**Why not just look at feature homophily?** The graph-divergence experiment
([`scripts/experiments/graph_divergence/`](../graph_divergence/README.md)) found
the edge-vs-random feature-cosine gap is small on every retweet graph
(0.015–0.070) — features aren't *smooth over edges*. But low homophily ≠ useless
features: a feature can be predictive without being graph-smooth. This
experiment measures **usefulness** directly, per task, via intervention.

## Method

For a fixed checkpoint, evaluate each task under four feature conditions at the
same regime (n-way / shots / split), then compare accuracy. The three ablations
differ along two axes — *node distinctness* and *real neighborhood content* —
which is what lets them separate competing explanations:

- `zero`    — replace every node feature with zeros (`AblateAllFeatures("zero")`).
  Removes distinctness *and* content.
- `permute` — permute feature rows across nodes *within each subgraph*
  (`AblateAllFeatures("permute")`): keeps distinctness, keeps the subgraph's real
  feature multiset, only breaks the node↔feature binding.
- `noise`   — resample every node's features from the *full graph's* feature
  distribution (`RandomNodeAttr`, token `NR1.0`): keeps distinctness and stays
  in-distribution, but destroys the *real neighborhood content*.

| condition | node distinct? | node↔feature binding | real neighborhood content |
|-----------|:--:|:--:|:--:|
| intact  | ✓ | correct  | ✓ |
| permute | ✓ | scrambled | ✓ (preserved) |
| noise   | ✓ | n/a       | ✗ (destroyed) |
| zero    | ✗ | —         | ✗ |

**Why the extra `noise` condition matters.** `permute` alone is *not* enough to
conclude a task ignores content: permute preserves the subgraph's feature
multiset, so a model that uses content as a permutation-invariant *bag* is also
permute-invariant. `noise` breaks that tie — if accuracy holds under `noise`,
the model only needs distinct vectors (features-as-distinguishers); if it
collapses under `noise` but survives `permute`, the model uses the *real
neighborhood content* (features-as-content).

## Mechanism (what was added)

- `data/augment.py`: `AblateAllFeatures(mode={zero,permute})` (tokens `FZ`/`FP`);
  `noise` reuses the existing `RandomNodeAttr` (token `NR1.0`, full-graph resample).
  Unit tests: `data/tests/test_ablate_features.py`.
- `experiments/params.py`: `--ablate_features {none,zero,permute,noise}` — composes
  the token into `--augmentation` and forces `--augment_test True`. Intended for
  `-eval_only True` runs; warns otherwise.
- `scripts/experiments/eval/eval_ckpts_all_graph_tasks_tucker.py`:
  `--ablate-features` pass-through; ablated runs get a `_ablZ` / `_ablP` / `_ablN`
  tag so they don't collide with intact runs.

## Run (Tucker)

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate prodigy

# 1. list the checkpoint(s) to probe, one per line: "<model_name> <ckpt.pt>"
cat > scripts/experiments/feature_ablation/model_list.txt <<'LIST'
nm_pretrained  /dataMeR1/phil/gfm/prodigy/log/<your_nm_run>/models/<ckpt>.pt
LIST

# 2. sweep intact/zero/permute over the tasks (forwards extra args to the driver)
MODEL_LIST=scripts/experiments/feature_ablation/model_list.txt \
  bash scripts/experiments/feature_ablation/run_feature_ablation_tucker.sh \
  --datasets midterm,covid19_twitter,ukr_rus_twitter,twibot20 \
  --tasks neighbor_matching,temporal_link_prediction,classification \
  --shots 3 --nm-n-way 30 --gpus 0

# 3. collect the intact-vs-ablated gap table
python3 scripts/experiments/feature_ablation/parse_feature_ablation.py \
  --log-root log --out scripts/experiments/feature_ablation/feature_ablation_results.csv
```

The intact (`none`) pass reuses the standard eval path, so if matching intact
runs already exist in `log/` you can skip `none` via `MODES="zero permute"`.

## Results & findings

See **[`FINDINGS.md`](FINDINGS.md)** for the full writeup (executive summary,
per-treatment hypotheses, results tables, and evidence-based takeaways).

Headline: **NM relies on the real feature *content* of a node's neighborhood, not
topology or mere distinctness** — `noise` collapses NM to chance like `zero`, while
`permute` is harmless. Raw data:
[`feature_ablation_results.csv`](feature_ablation_results.csv),
[`feature_label_probe_results.csv`](feature_label_probe_results.csv),
[`feature_only_nm_results.csv`](feature_only_nm_results.csv).

## Files

- `run_feature_ablation_tucker.sh` — loops modes (`MODES=`, default
  `none zero permute`; add `noise`) over the driver.
- `parse_feature_ablation.py` — pairs intact↔ablated run dirs, writes the gap CSV.
- `feature_label_probe.py` — feature→label logreg (feature-quality check).
- `feature_only_nm_probe.py` — prototype-NN NM in raw feature space (no model).
- `feature_ablation_results.csv`, `feature_label_probe_results.csv`,
  `feature_only_nm_results.csv` — results.
- `model_list.txt` — (you create) checkpoints to probe.

## Caveats

- **GPU job.** Each pass loads the checkpoint and runs forward passes — light,
  but it shares GPUs with anything else running. Schedule accordingly.
- **Run all three ablations.** `permute` alone is not interpretable in isolation
  (it preserves the content bag). Always pair it with `noise` to separate
  features-as-distinguishers from features-as-content, and `zero` as the blunt
  cross-check.
- **Same regime.** Only compare intact vs. ablated at identical n-way/shots/split
  — the parser pairs by run-dir config key, which encodes these.
- **Complement, not replacement.** For labeled graphs, the per-node logistic
  probe (`data/data/<ds>/scripts/logistic_regression_baseline.py`) answers the
  downstream-feature question directly; this ablation covers NM/LP too.
