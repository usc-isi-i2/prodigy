# scripts

SLURM job submission scripts and supporting utilities for training and evaluating PRODIGY models across all datasets. All dataset-specific values live in `config/` — the submission scripts themselves are dataset-agnostic.

## Layout

```
scripts/
  config/
    midterm.sh              # dataset-specific values for midterm
    ukr_rus_twitter.sh      # dataset-specific values for ukr_rus_twitter
    covid19_twitter.sh      # dataset-specific values for covid19_twitter
  train_single_task.sbatch  # generic training job (train1 and train2)
  eval_model_list.sbatch    # generic model-list evaluation job
  submit_train1.sh          # submit a from-scratch single-task training run
  submit_train2.sh          # submit all cross-task fine-tuning runs
  submit_eval.sh            # submit a model-list evaluation run
  submit_eval_midterm_to_ukr_rus_all_tasks.sh   # cross-dataset eval wrapper
  submit_eval_ukr_rus_to_midterm_all_tasks.sh   # cross-dataset eval wrapper
  dataset_graph_summary.py  # print a TSV/JSON summary row for a graph artifact
  plotting/                 # Jupyter notebooks for result plots
  *.txt                     # model list files used by submit_eval.sh
```

---

## Workflow overview

```
Train1 (from scratch, one task)
  submit_train1.sh <dataset> <task>
        │
        └─► train_single_task.sbatch  (sources config/<dataset>.sh)

Train2 (cross-task fine-tune, all 6 combinations)
  submit_train2.sh <dataset> <nm_ckpt> <lp_ckpt> <pl_ckpt>
        │
        └─► train_single_task.sbatch × 6  (with CKPT_PATH set)

Combo / leave-one-out (fine-tune on 2 tasks, eval on 3rd)
  submit_train_combo_all.sh <dataset> <nm_ckpt> <lp_ckpt> <pl_ckpt> [shots_csv]
        │
        └─► train_combo.sbatch × 6   (both orderings per held-out task)
              Phase 1: fine-tune src checkpoint on second task
              Phase 2: eval combo checkpoint on held-out task

Eval (all tasks × all shot counts from a model list)
  submit_eval.sh <dataset> <model_list.txt> [shots_csv]
        │
        └─► eval_model_list.sbatch  (sources config/<dataset>.sh)
```

---

## Configuration files (`config/`)

Each dataset has one config file that defines everything environment-specific. The sbatch templates source the relevant config at runtime based on the `$DATASET` environment variable.

| Variable | Purpose |
|---|---|
| `DATASET` | Dataset name passed to `run_single_experiment.py` |
| `GRAPH_ROOT` | Directory containing `.pt` graph artifacts |
| `LOG_DIR` | SLURM stdout/stderr destination |
| `SLURM_MEM` | Memory override passed to `sbatch --mem` |
| `WORKERS` | DataLoader worker count |
| `FEATURE_SUBSET` | Value for `--midterm_feature_subset` |
| `INPUT_DIM` | Embedding dimension |
| `EDGE_VIEW` | Value for `--midterm_edge_view` (unset for midterm → framework default) |
| `SUPPORTED_TASKS` | Bash array of tasks this dataset supports |
| `NM_GRAPH`, `LP_GRAPH`, `PL_GRAPH` | Graph filename per task |
| `NM_N_WAY`, `NM_N_QUERY`, `NM_TRAIN_SHOTS` | Task-level hyperparameters (repeated for `LP_`, `PL_`) |
| `LP_TARGET_EDGE_VIEW` | `--midterm_target_edge_view` for link prediction |
| `PL_LABEL_DOWNSAMPLE` | `--midterm_label_downsample` for classification |
| `PL_DATASET_LEN_CAP` | `--dataset_len_cap` for classification |

> **Note on `--midterm_*` flags:** these are the CLI argument names defined in `experiments/run_single_experiment.py`. Despite the prefix they apply to all datasets.

---

## Submission scripts

### `submit_train1.sh` — train from scratch

```bash
submit_train1.sh <dataset> <task> [extra_sbatch_args...]
```

| Argument | Values |
|---|---|
| `dataset` | `midterm` \| `ukr_rus_twitter` \| `covid19_twitter` |
| `task` | `nm` \| `lp` \| `pl` (or full names) |
| `extra_sbatch_args` | Forwarded verbatim to `sbatch`, e.g. `--time=02:00:00` |

**Examples:**

```bash
# Train midterm neighbor-matching from scratch
./submit_train1.sh midterm nm

# Train ukr_rus link-prediction, with a longer time limit
./submit_train1.sh ukr_rus_twitter lp --time=03:00:00

# Train covid19 neighbor-matching
./submit_train1.sh covid19_twitter nm
```

The job name is `train1_<dataset>_<task>` and output goes to `$LOG_DIR/%x_%j.out`.

---

### `submit_train2.sh` — cross-task fine-tuning

Takes three train1 checkpoints (one per task) and submits six sbatch jobs — each source model fine-tuned on its two non-training tasks.

```bash
submit_train2.sh <dataset> <nm_ckpt> <lp_ckpt> <pl_ckpt>
```

Pass an empty string `""` for any checkpoint that does not exist (e.g. covid19_twitter has no `pl` checkpoint):

```bash
./submit_train2.sh covid19_twitter \
  state/train1_covid19_twitter_nm_.../state_dict \
  state/train1_covid19_twitter_lp_.../state_dict \
  ""
```

The six submitted jobs are:

| Source | → Target |
|---|---|
| nm | lp, pl |
| lp | nm, pl |
| pl | nm, lp |

---

### `submit_train_combo.sh` / `submit_train_combo_all.sh` — leave-one-out combo experiments

Trains on two tasks sequentially (fine-tune an existing checkpoint on a second task) then evaluates the resulting checkpoint on the held-out third task. Both phases run in a single SLURM allocation.

**Single combination:**

```bash
submit_train_combo.sh <dataset> <src_task> <src_ckpt> <finetune_task> <eval_task> [shots_csv]
```

```bash
# Fine-tune the midterm NM checkpoint on LP, then eval on PL
./submit_train_combo.sh midterm \
  nm state/train1_midterm_nm_.../state_dict \
  lp pl
```

**All 6 leave-one-out combinations at once:**

```bash
submit_train_combo_all.sh <dataset> <nm_ckpt> <lp_ckpt> <pl_ckpt> [shots_csv]
```

```bash
# Submit all 6 midterm combos
./submit_train_combo_all.sh midterm \
  state/train1_midterm_nm_.../state_dict \
  state/train1_midterm_lp_.../state_dict \
  state/train1_midterm_pl_.../state_dict

# covid19_twitter has no pl checkpoint — pass empty string
./submit_train_combo_all.sh covid19_twitter \
  state/train1_covid19_twitter_nm_.../state_dict \
  state/train1_covid19_twitter_lp_.../state_dict \
  ""
```

The full leave-one-out matrix (6 jobs):

| Held-out (eval) | Training order |
|---|---|
| pl | nm → fine-tune lp → eval pl |
| pl | lp → fine-tune nm → eval pl |
| lp | nm → fine-tune pl → eval lp |
| lp | pl → fine-tune nm → eval lp |
| nm | lp → fine-tune pl → eval nm |
| nm | pl → fine-tune lp → eval nm |

Unsupported tasks (e.g. `pl` on covid19_twitter) are skipped automatically with a warning. Each job runs up to 8 hours (fine-tune ≈ 2h + eval across all shots ≈ 4h).

**How the checkpoint is located between phases:**
After Phase 1 completes, the job searches `state/combo_<dataset>_<src>_ft_<ft>_*/checkpoint/*.ckpt` for the most recently written checkpoint and passes it directly to Phase 2. No manual intervention is needed.

---

### `submit_eval.sh` — model-list evaluation

```bash
submit_eval.sh <dataset> <model_list.txt> [shots_csv]
```

| Argument | Default | Notes |
|---|---|---|
| `dataset` | — | Determines which graph and tasks to evaluate on |
| `model_list.txt` | — | One checkpoint per line (see format below) |
| `shots_csv` | `0,1,2,5,10` | Comma-separated shot counts |

The job runs every checkpoint × every supported task × every shot count in a single sbatch allocation. All tasks listed in `SUPPORTED_TASKS` for the chosen dataset are evaluated automatically.

**Model list format** — one of:
```
# This is a comment
state/train1_midterm_nm_.../state_dict
nm  state/train1_midterm_nm_.../state_dict
```
Either bare path (model name inferred from parent directory) or `<name> <path>`.

**Examples:**

```bash
# Evaluate all models in eval1_model_list.txt on midterm, default shots
./submit_eval.sh midterm scripts/eval1_model_list.txt

# Same models, custom shot list
./submit_eval.sh midterm scripts/eval1_model_list.txt 0,1,5

# Evaluate covid19-trained models on the covid19 graph
./submit_eval.sh covid19_twitter scripts/eval1_covid_model_list.txt
```

---

### Cross-dataset evaluation

To evaluate models trained on one dataset against another dataset's graph, pass the **target** dataset to `submit_eval.sh` and a model list from the source training run:

```bash
# Midterm-trained models evaluated on ukr_rus_twitter
./submit_eval.sh ukr_rus_twitter scripts/midterm_train1_eval_on_ukr_rus_model_list.txt

# ukr_rus-trained models evaluated on midterm
./submit_eval.sh midterm scripts/ukr_rus_train1_eval_on_midterm_model_list.txt
```

The two convenience wrappers do exactly this:

```bash
./submit_eval_midterm_to_ukr_rus_all_tasks.sh [shots_csv]
./submit_eval_ukr_rus_to_midterm_all_tasks.sh [shots_csv]
```

---

## `dataset_graph_summary.py`

Prints a single TSV or JSON row comparing a graph artifact against its raw source data.

```bash
python3 scripts/dataset_graph_summary.py \
  --dataset midterm \
  --graph data/data/midterm/graphs/retweet_graph.pt \
  [--raw_glob "/path/to/csvs/*/*.csv"] \
  [--max_files 50] \
  [--format tsv|json]
```

If `--raw_glob` and `--max_files` are omitted, they are inferred from the `.meta.json` sidecar next to the graph file.

Output columns: `dataset`, `n_tweets`, `n_users`, `n_nodes`, `n_edges`, `mean_deg`, `n_node_features`, `n_edge_features`, `mean_centrality`.

---

## Adding a new dataset

1. Add `config/<dataset>.sh` following the pattern of the existing configs. Set `SUPPORTED_TASKS` to exclude tasks the dataset cannot support.
2. Add the dataset's preprocessing scripts under `data/data/<dataset>/scripts/` (see `rapids/README.md`).
3. `submit_train1.sh`, `submit_train2.sh`, and `submit_eval.sh` require no changes — they discover the dataset through the config file.

---

## Resource defaults

| Dataset | Memory | Notes |
|---|---|---|
| midterm | 32 GB | Smaller CSV files |
| ukr_rus_twitter | 64 GB | Large interleaved CSVs |
| covid19_twitter | 64 GB | Large JSON files |

These are set in each `config/<dataset>.sh` as `SLURM_MEM` and applied via `sbatch --mem=` at submission time, overriding the conservative 64 GB fallback in the sbatch templates.
