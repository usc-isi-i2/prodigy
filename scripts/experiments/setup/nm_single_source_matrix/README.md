# NM single-source transfer matrix (8×8)

**Question.** Train a neighbor-matching (NM) model on **each single graph alone**,
then evaluate every model on **every graph** — an 8×8 `train × test` NM-AUC matrix.
The **diagonal** is each graph's in-domain specialist; the **off-diagonal** is
zero-training-overlap transfer. This is the single-source counterpart to the merged
NM "interpolation ladder" (`scripts/experiments/setup/covid_ukr`, plotted in
`scripts/experiments/analysis/transfer/ladders/prodigy_nm/canonical/nm_ladder/data/nmladder_results.csv`): the ladder trains on a *growing
merge*, this trains on *one source at a time*, and both are read on the same 8 columns.

## The 8 graphs

`ukr_rus_twitter`, `covid19_twitter`, `midterm`, `covid_political`, `election2020`,
`ukr_rus_suspended`, `twibot20`, `cp_hk_twitter` — the same 8 single-source graphs
that form the merged ladder's columns (and the all-8 rung's inputs).

## Protocol — held fixed except the training source

All 8 configs are byte-identical except `dataset` / `root` / `graph_filename` /
`prefix` / `tags`:

- **Architecture / objective:** argparse defaults — **no augmentation, no
  attr-regression**, default `emb_dim` / `layers` / `dropout` / LR. Same plain arch
  as `nm_transfer_matrix/ukr_nm.yaml` and the ladder's `ukr_only_nm.yaml`.
- **Task / sampling:** `neighbor_matching`, `n_way: 30`, `n_shots: 3`, `n_query: 4`,
  `n_hop: 1`, `edge_view: default`, `feature_subset: all` (768-dim), `seed: 0`.
- **Budget — matched-40k:** `epochs: 5 × dataset_len_cap 10000 = 50,000` steps, with
  `checkpoint_step: 10000` (ckpts at 10k/20k/30k/40k). Eval reads **`state_dict_40000`**
  — the same matched budget the ladder was evaluated at (the trainer skips the
  final-step ckpt, so 50k > 40k guarantees a 40k ckpt exists). `eval_step: 100000`
  skips periodic val-eval (~2× faster to 40k).

Single-source ⇒ no cross-source sampling knobs needed (each episode is already
within one graph), so this is directly comparable to the ladder's within-source rungs.

**Cross-check built in:** the `ukr → ukr/covid/midterm` row should reproduce ladder
rung 1 (`.948 / .973 / .874`) and `nm_transfer_matrix`'s single-ukr
(`.9497 / .9741 / .8840`). `build_ss_matrix.py` prints the ukr→ukr diagonal and flags
drift from `.948`.

## Files

| File | Purpose |
|------|---------|
| `<dataset>.yaml` (×8) | The 8 training configs (identical except the data source). |
| `train_nm_tucker.sh` | Train one source: `./train_nm_tucker.sh midterm.yaml`. `DRY_RUN=1` to preview. |
| `run_all_train_tucker.sh` | Train all 8 across a GPU pool (round-robin, each GPU sequential). |
| `make_model_list.sh` | Write `model_list.txt` → each run's `state_dict_40000.ckpt`. |
| `eval_matrix_tucker.sh` | Eval every model on all 8 graphs (NM 30-way 3-shot). |
| `build_ss_matrix.py` | Aggregate eval logs → 8×8 matrix + `nm_single_source_matrix.csv`. |

Eval reuses `scripts/eval/eval_ckpts_all_graph_tasks_tucker.py`; all 8
graphs are already in its dataset registry, so no code change is needed.

## How to run (on Tucker, `prodigy` conda env)

```bash
cd scripts/experiments/nm_single_source_matrix

# 0. (recommended) smoke-test the two SMALL graphs first — see caveat below.
DRY_RUN=1 ./run_all_train_tucker.sh                 # preview all 8 commands
# EPOCHS override via extra arg: ./train_nm_tucker.sh election2020.yaml --epochs 1

# 1. Train all 8 single-source models. Default GPUs 0-3 (2 runs/GPU); give more
#    GPUs to go faster (one per config = fastest).
GPUS="0 1 2 3" ./run_all_train_tucker.sh            # logs in ./run_logs/

# 2. Point model_list.txt at each run's matched-40k checkpoint.
STATE_DIR=/dataMeR1/phil/gfm/prodigy/state ./make_model_list.sh

# 3. Evaluate every model on every graph (64 NM jobs; fan out over GPUs).
GPUS="0,1,2,3" ./eval_matrix_tucker.sh

# 4. Build the 8×8 matrix + CSVs.
python3 build_ss_matrix.py --log-root /dataMeR1/phil/gfm/prodigy/log
```

`DATA_ROOT=` overrides the graph root for eval (defaults to `/dataMeR1/phil/data` to
match the training configs / ladder; the shared harness otherwise defaults to
`/dataMeR2`). `SKIP="ukr_rus_twitter.yaml"` skips configs — e.g. reuse the ladder's
existing ukr `state_dict_40000` instead of retraining it (add that path to
`model_list.txt` by hand).

## Reading the result

`nm_single_source_matrix.csv` has one row per train source and the 8 canonical test
columns (roc_auc) — same shape as `nmladder_results.csv`, so you can stack the
single-source rows under the merged-ladder rows for a combined figure.

- **Diagonal** (`X → X`) — in-domain specialist ceiling per graph.
- **Off-diagonal** (`X → Y`) — pure transfer with zero training overlap. Compare a
  column against the ladder: does *any* single specialist beat the merged model that
  was trained on that graph? Does merging ever beat the best specialist?

## Caveats

- **1 seed** (`seed: 0`), like the ladder and `nm_transfer_matrix`. Sub-1% AUC gaps
  are within run-to-run noise; the configs take `--seed N` for a multi-seed follow-up.
- **Small/sparse graphs.** `election2020` and `ukr_rus_suspended` are small — the eval
  harness drops them to `nm_n_query=1`. 30-way NM *training* episodes need centers with
  enough neighbors; if either run errors or its diagonal is degenerate, smoke-test with
  `--epochs 1` and consider a smaller `n_way` for those two (at the cost of exact
  comparability). This is likely why they were never trained single-source before.
- **Checkpoint budget.** Everything is read at `state_dict_40000`. If a small graph
  never reaches 40k steps (shouldn't happen at 50k configured), `make_model_list.sh`
  WARNs and skips it — check the warnings before building the matrix.
