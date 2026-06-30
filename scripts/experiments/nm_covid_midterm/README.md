# NM covid/midterm validation

Validates, on **covid + midterm**, the findings established on ukr/covid
(see `../nm_transfer_matrix` and `../nm_cross_source_shortcut`):
1. a fair merged model is **not worse** than single-source (no inversion), and
2. confining NM episodes to one source (no cross-source negatives) helps.

Everything is held fixed except the training data / sampling — plain default
architecture, no augmentation, 3-shot / 30-way eval, seed 0.

## The 5 training regimes

| config | prefix | what |
|---|---|---|
| `midterm_nm.yaml` | nm_cm_midterm | single source: midterm |
| `covid_nm.yaml` | nm_cm_covid | single source: covid |
| `merged_nm.yaml` | nm_cm_merged | merged-naive (proportional, mixed episodes) |
| `merged_within_nm.yaml` | nm_cm_within | merged, episodes confined to one source, source ∝ size |
| `merged_within_balanced_nm.yaml` | nm_cm_within_balanced | merged, episodes confined to one source, source uniform (50/50) |

Single-source budget 60k episodes (6×10k); merged budgets 120k (12×10k).

**The two within variants differ in one knob** —
`neighbor_sampling_episode_source_weighting`:
- `proportional`: P(source) ∝ node count → per-node center marginal matches the
  naive baseline; isolates *only* the cross-source-negative effect.
- `balanced`: P(source) uniform → also rebalances per-domain exposure (matters here
  because **midterm is only ~1.5% of the merge** — covid ≈ 23.0M nodes, midterm ≈
  0.34M, so under proportional sampling midterm is seen ~1.5% of episodes).

## Graph

Built by `scripts/graph_construction/merge_covid_midterm.yaml` →
`/dataMeR1/phil/data/merged/graphs/covid_midterm_retweet_graph.pt`
(disjoint merge, graph_id 0=covid, 1=midterm; 23.35M nodes, 768-dim features).

## Run (Tucker, prodigy env, tmux)

Only GPUs 0-3 are available; the 5 regimes are scheduled across them (the two fast
single-source runs share GPU 0). Run under tmux so it survives closing the laptop.

```bash
cd scripts/experiments/nm_covid_midterm

# (optional) smoke-test the balanced flag first — confirms the strata banner prints:
source /home/mhchu/miniconda3/etc/profile.d/conda.sh && conda activate prodigy
WANDB_MODE=offline ./train_nm_tucker.sh merged_within_balanced_nm.yaml --device 0 \
  --epochs 1 -ds_cap 20 -eval_step 20 -ckpt_step 20 --val_len_cap 5 --test_len_cap 5 --prefix nm_cm_smoke
#   look for: "Neighbor sampling graph_id strata (confine-to-one-source): covid:..., midterm:..."

# train all 5 across GPUs 0-3 (parallel lanes; survives laptop close):
tmux new -s cm
./run_all_train_tucker.sh    # GPU0: midterm->covid | GPU1: merged | GPU2: within | GPU3: within-balanced
#   detach: Ctrl-b d   reattach: tmux attach -t cm   watch: tail -f run_logs/*.log

# after training: eval all 5 on covid + midterm + merged (3-shot, 30-way), then tables.
STATE_DIR=/dataMeR1/phil/gfm/prodigy/state ./make_model_list.sh
cat model_list.txt
./eval_tucker.sh --device 0 --continue-on-error
python3 build_matrix.py --log-root /dataMeR1/phil/gfm/prodigy/log \
  --shots 3 --n-way 30 --metric all --out-csv matrix.csv
```

`build_matrix.py` prints accuracy, f1, and AUC tables (train rows × test cols:
midterm / covid / merged). Reuse `--metric accuracy` etc. for a single table.

> Reminder: NM is degenerate at 0-shot (chance / AUC≈0.5) — always eval at shots ≥ 3.
