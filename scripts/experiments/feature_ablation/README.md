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

For a fixed checkpoint, evaluate each task twice more than usual — features
**intact**, **zeroed**, and **permuted** — at the same regime (n-way / shots /
split), then compare accuracy:

- `zero`    — replace every node feature with zeros (`AblateAllFeatures("zero")`).
- `permute` — permute feature rows across nodes within each subgraph
  (`AblateAllFeatures("permute")`): preserves the subgraph's feature
  distribution but destroys the node↔feature alignment, so any residual accuracy
  must come from topology. This is the cleaner ablation (keeps input stats);
  `zero` is the blunt version and a useful cross-check.

**Read the gap** `intact − ablated`:
- large gap → the task genuinely uses features;
- ≈ 0 gap → the task rides on topology, features aren't contributing.

Comparing the gap across tasks shows *where* features matter — in particular
whether NM, our main pretraining task, is feature-driven or topology-driven.

## Mechanism (what was added)

- `data/augment.py`: `AblateAllFeatures(mode={zero,permute})` + `get_aug` tokens
  `FZ` / `FP`. Unit tests: `data/tests/test_ablate_features.py`.
- `experiments/params.py`: `--ablate_features {none,zero,permute}` — composes the
  token into `--augmentation` and forces `--augment_test True`. Intended for
  `-eval_only True` runs; warns otherwise.
- `scripts/experiments/eval/eval_ckpts_all_graph_tasks_tucker.py`:
  `--ablate-features` pass-through; ablated runs get a `_ablZ` / `_ablP` tag so
  they don't collide with intact runs.

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

## Files

- `run_feature_ablation_tucker.sh` — loops `none/zero/permute` over the driver.
- `parse_feature_ablation.py` — pairs intact↔ablated run dirs, writes the gap CSV.
- `model_list.txt` — (you create) checkpoints to probe.

## Caveats

- **GPU job.** Each pass loads the checkpoint and runs forward passes — light,
  but it shares GPUs with anything else running. Schedule accordingly.
- **`permute` vs `zero`.** Prefer `permute` as the headline ablation; report
  `zero` alongside. If they disagree sharply, the model may be keying on feature
  *norm/scale* (which `zero` removes but `permute` keeps) rather than identity.
- **Same regime.** Only compare intact vs. ablated at identical n-way/shots/split
  — the parser pairs by run-dir config key, which encodes these.
- **Complement, not replacement.** For labeled graphs, the per-node logistic
  probe (`data/data/<ds>/scripts/logistic_regression_baseline.py`) answers the
  downstream-feature question directly; this ablation covers NM/LP too.
