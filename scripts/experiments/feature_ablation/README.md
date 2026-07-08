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
neighborhood content* (features-as-content). (This corrects an earlier reading
that treated permute-invariance as proof of content-disuse — see Findings.)

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

## Findings

> **Correction.** An earlier version of this file concluded "NM ignores node-feature
> content; the encoder strips it." That was based on `permute`-invariance alone and
> is **wrong** — the `noise` condition (added later) overturns it. NM relies heavily
> on real feature content. The corrected story is below.

**Phase A — NM (30-way, `nm_matrix_covid`)** → `feature_ablation_results.csv`
(accuracy; chance = 1/30 ≈ 0.033):

| dataset | intact | zero | permute | noise |
|---|---|---|---|---|
| covid19_twitter (in-domain) | 0.664 | 0.073 | 0.626 | **0.061** |
| midterm | 0.313 | 0.086 | 0.311 | **0.064** |
| twibot20 | 0.406 | 0.066 | 0.407 | **0.058** |

`noise` collapses NM to ~chance, essentially identical to `zero`, while `permute`
is harmless. Since `noise` keeps nodes perfectly distinct and only destroys the
*real* neighborhood content, **distinctness is not what NM needs — real feature
content is.** And both `zero` and `noise` leave topology intact yet give chance,
so **topology alone ≈ chance for this model.** NM matches a query to a center via
their shared neighborhood feature signature (a permutation-invariant content
*bag*, since `permute` is fine), not via topology and not via mere distinctness.

**Feature-quality probe** (`feature_label_probe.py` → `feature_label_probe_results.csv`).
Logistic regression from raw features to the node label (no graph): AUC 0.95
(election2020), 0.91 (covid_political), 0.71 (twibot20), 0.60 (ukr_rus_suspended).
The bio embeddings carry strong, linearly-decodable signal — features are good.

**Feature-only NM probe** (`feature_only_nm_probe.py` → `feature_only_nm_results.csv`).
Prototype nearest-neighbor in raw feature space (no model), 30-way / 3-shot:
twibot20 real 0.169 (AUC 0.66) vs permuted 0.035; midterm real 0.103 (AUC 0.61)
vs permuted 0.032; chance 0.033. Neighborhoods *are* feature-distinguishable at
the community level (despite low edge homophily), which is exactly the signal the
full model exploits (and extracts better: 0.41 / 0.31).

**Phase B — classification linear-probe on the frozen NM representation** (20-shot),
`zero`/`permute` only (no `noise` yet): election2020 intact/zero/permute AUC
0.979/0.503/0.978; covid_political 0.912/0.613/0.911; twibot20 0.680/0.715/0.673.
Permute-invariant on every graph — but **by the correction above, permute-invariance
does NOT imply the encoder discards content** (permute preserves the content bag).
So the earlier "encoder strips content" claim is **not established**; it needs a
`noise` re-run. The twibot20 gap (rep 0.680 < raw-feature logreg 0.707) remains a
*tentative* hint that individual-node semantics may be under-encoded relative to
neighborhood-aggregate content — but that is unconfirmed.

**Implication (revised).** Features are good, and NM already **uses** feature
content heavily — the original "NM ignores features, so add a feature-content
task" motivation is overturned. Topology, not features, is the underused channel
here. The remaining open question is narrower: does the encoder preserve
*individual-node* bio semantics (needed for low-homophily downstream like
twibot20), or only the *neighborhood-aggregate* signature it uses for NM? Resolve
with (1) `noise` on the Phase-B classification probe and (2) a feature-reconstruction
probe (predict a node's raw bio from its frozen representation).

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
  cross-check. The earlier permute-only conclusion here was wrong for exactly
  this reason.
- **Same regime.** Only compare intact vs. ablated at identical n-way/shots/split
  — the parser pairs by run-dir config key, which encodes these.
- **Complement, not replacement.** For labeled graphs, the per-node logistic
  probe (`data/data/<ds>/scripts/logistic_regression_baseline.py`) answers the
  downstream-feature question directly; this ablation covers NM/LP too.
