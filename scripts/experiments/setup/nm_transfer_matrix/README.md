# NM transfer matrix — fair single-vs-merged validation

## Motivation

An earlier run produced a counter-intuitive result: a neighbor-matching (NM)
model **trained on UKR** scored a *higher* NM AUC **on COVID** than a model
trained on the **merged** UKR+COVID graph — and symmetrically, COVID→UKR beat
merged→UKR. This experiment checks whether that inversion is real or an artifact
of an unfair comparison.

It almost certainly *was* partly unfair: the original single-source configs
(`ukr_only/ukr_only_nm.yaml`, `covid_only/covid_only_nm.yaml`) use the **plain
default** architecture with no augmentation, while the merged run that was
compared against them (`covid_ukr/merged_ukr_rus_covid_nm_aug.yaml`) used a
larger model (`emb_dim: 512`, `layers: S2,U,M2`), dropout, a tuned LR, `NZ0.3`
augmentation, and `attr_regression_weight: 1000`. So the three models differed in
**architecture, augmentation, and training objective** — not just the data
source. Any of those could drive the inversion.

This folder re-runs the comparison with **everything held fixed except the
training data**, then evaluates every model on every test domain to get a clean
train × test AUC matrix.

## What is held fixed

All three training runs share identical settings (see the three YAMLs, which are
byte-for-byte identical except for the data source and the merged epoch count):

- **Architecture / objective:** plain argparse defaults — **no augmentation**,
  **no attr-regression**, default `emb_dim` / `layers` / `dropout` / LR.
- **Task / sampling:** `neighbor_matching`, `n_way: 30`, `n_shots: 3`,
  `n_query: 4`, `n_hop: 1`, `edge_view: default`, `original_features: true`,
  `seed: 0`.
- **Per-epoch episodes:** `dataset_len_cap: 10000`.

### Compute budget — fixed *per-domain exposure* (not fixed total)

- Single-source (`ukr_nm`, `covid_nm`): **6 epochs × 10k = 60,000 episodes**.
- Merged (`merged_nm`): **12 epochs × 10k = 120,000 episodes** (2×), so each
  domain is seen roughly as many times as in its own single-source run.

**Caveat:** the merged graph is the existing **proportional ("as-is")** merge,
which is COVID-dominant. Under uniform center sampling, doubling merged episodes
still **over-exposes COVID and under-exposes UKR** relative to their
single-source runs — exposure is only exactly matched if the two domains are
equal size. A size-balanced merge would fix this but was intentionally left out
of this experiment. Keep this in mind when reading the off-diagonal cells.

This is **1 seed** per source. A single AUC per cell cannot separate a real
effect from run-to-run variance — if the inversion reproduces here, the natural
follow-up is to re-run with ≥3 seeds for error bars.

## Files

| File | Purpose |
|------|---------|
| `ukr_nm.yaml`, `covid_nm.yaml`, `merged_nm.yaml` | The three training configs (identical except data source + merged epochs). |
| `train_nm_tucker.sh` | Train one source: `./train_nm_tucker.sh ukr_nm.yaml`. |
| `run_all_train_tucker.sh` | Train all three **in parallel, one per GPU** (logs to `run_logs/`). |
| `make_model_list.sh` | After training, auto-write `model_list.txt` pointing at each run's final checkpoint. |
| `model_list.txt` | `<train_source> <final_checkpoint_path>`, one per line (generated). |
| `eval_nm_matrix_tucker.sh` | Eval all 3 models on all 3 test domains (NM, zero-shot). |
| `build_auc_matrix.py` | Aggregate eval logs into the train × test AUC matrix (stdlib only). |

The eval reuses `scripts/eval/eval_ckpts_all_graph_tasks_tucker.py`.
A `merged_ukr_rus_covid` entry was added to that script's dataset registry so the
merged graph can also be a **test** target (the 3rd matrix column).

## How to run (on Tucker, in the `prodigy` conda env)

```bash
cd scripts/experiments/nm_transfer_matrix

# 1. Train all three sources (60k / 60k / 120k episodes), one per GPU.
#    Default: ukr->gpu0, covid->gpu1, merged->gpu2. Override with GPUS="a b c".
#    Per-run logs go to ./run_logs/. Note: merged (120k) takes ~2x as long as
#    the single-source runs, so it finishes last.
./run_all_train_tucker.sh                 # or: GPUS="2 3 5" ./run_all_train_tucker.sh

# 2. Point model_list.txt at the final checkpoints.
STATE_DIR=/dataMeR1/phil/gfm/prodigy/state ./make_model_list.sh

# 3. Evaluate every model on every test domain (NM AUC, zero-shot, ROC saved).
./eval_nm_matrix_tucker.sh

# 4. Build the matrix.
python3 build_auc_matrix.py \
  --log-root /dataMeR1/phil/gfm/prodigy/log \
  --out-csv auc_matrix.csv
```

Use `DRY_RUN=1 ./train_nm_tucker.sh ukr_nm.yaml` to print commands without
running, and `--dry-run` on `eval_nm_matrix_tucker.sh` to preview eval jobs.

## Reading the result

`build_auc_matrix.py` prints a `train` (rows) × `test` (cols) AUC matrix over
`{ukr, covid, merged}` and explicitly reports the cells of interest:

```
test=covid: single(ukr)=... vs merged=...  -> INVERSION reproduced / no inversion
test=ukr:   single(covid)=... vs merged=... -> ...
```

- **Diagonal** (e.g. ukr→ukr) = in-domain NM quality.
- **Off-diagonal single→other** vs **merged→other** is the original question:
  does single-source still beat the merged model cross-domain when architecture,
  augmentation, and objective are identical? If yes, the effect survives the fair
  comparison and is about *data mixture*, not the confounds above.
