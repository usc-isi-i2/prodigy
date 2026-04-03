# Cross-Dataset Transfer Evaluation

This guide explains how to train a model on one Twitter dataset and evaluate it on another — the standard way to test out-of-distribution generalization in this repo.

## Overview

The full flow is:

1. **Train** on a source dataset (e.g. Midterm) using the normal training scripts.
2. **Record** the checkpoint path(s) in a model list file.
3. **Run** the eval sbatch script, pointing it at the model list and the target dataset.

The eval script loads each checkpoint, freezes the model weights (`--eval_only True`), and runs inference on the target dataset across all three tasks (neighbor matching, temporal link prediction, political leaning classification) and across multiple shot counts (default: 0, 1, 2, 5, 10).

---

## Step 1: Train on the source dataset

Use the standard training scripts (see [README.md](README.md)):

```bash
# Train on Midterm
sbatch scripts/submit_train1_midterm_lp.sh
sbatch scripts/submit_train1_midterm_nm.sh
sbatch scripts/submit_train1_midterm_pl.sh

# Train on Ukraine-Russia
sbatch scripts/submit_train1_ukr_rus_twitter_lp.sh
sbatch scripts/submit_train1_ukr_rus_twitter_nm.sh
sbatch scripts/submit_train1_ukr_rus_twitter_pl.sh

# Train on COVID-19
sbatch scripts/submit_train1_covid19_twitter_lp.sh
sbatch scripts/submit_train1_covid19_twitter_nm.sh
```

Checkpoints are saved to `state/<PREFIX>_<timestamp>/checkpoint/`.

---

## Step 2: Create or update a model list file

A model list file is a plain text file where each non-blank, non-comment line identifies one checkpoint to evaluate. Two formats are supported:

```
# Format 1: checkpoint path only (model name derived from parent directory)
state/train1_midterm_lp_24_03_2026_12_11_22/checkpoint/state_dict_7000.ckpt

# Format 2: explicit model name + checkpoint path (recommended)
lp state/train1_midterm_lp_24_03_2026_12_11_22/checkpoint/state_dict_7000.ckpt
nm state/train1_midterm_nm_24_03_2026_12_15_40/checkpoint/state_dict_1000.ckpt
pl state/train1_midterm_pl_24_03_2026_12_15_40/checkpoint/state_dict_2000.ckpt
```

The model name appears in W&B run prefixes and output filenames as `trained_on_<model_name>_eval_on_<task>_<shots>_shot`.

Existing model list files for reference:

| File | Source training | Target eval dataset |
|---|---|---|
| `scripts/midterm_train1_eval_on_ukr_rus_model_list.txt` | Midterm | Ukraine-Russia |
| `scripts/ukr_rus_train1_eval_on_midterm_model_list.txt` | Ukraine-Russia | Midterm |
| `scripts/ukr_rus_train1_eval_model_list.txt` | Ukraine-Russia | Ukraine-Russia (in-domain) |
| `scripts/eval1_model_list.txt` | Midterm | COVID-19 |

---

## Step 3: Submit the eval job

Pick the sbatch script that corresponds to the **target** dataset you want to evaluate on:

| Target dataset | sbatch script |
|---|---|
| Midterm | `scripts/eval_midterm_model_list_all_tasks.sbatch` |
| Ukraine-Russia | `scripts/eval_ukr_rus_twitter_model_list_all_tasks.sbatch` |
| COVID-19 | `scripts/eval_covid19_twitter_model_list_all_tasks.sbatch` |

Pass your model list file as the first argument. The second argument (optional) is a comma-separated list of shot counts:

```bash
# Midterm-trained models → evaluate on Ukraine-Russia (default shots: 0,1,2,5,10)
sbatch scripts/submit_eval_midterm_to_ukr_rus_all_tasks.sh

# Same, but only 1- and 5-shot
sbatch scripts/eval_ukr_rus_twitter_model_list_all_tasks.sbatch \
  scripts/midterm_train1_eval_on_ukr_rus_model_list.txt \
  1,5

# Ukraine-Russia-trained models → evaluate on Midterm
sbatch scripts/submit_eval_ukr_rus_to_midterm_all_tasks.sh

# Any model list → evaluate on COVID-19
sbatch scripts/eval_covid19_twitter_model_list_all_tasks.sbatch \
  scripts/eval1_model_list.txt \
  0,1,2,5,10
```

The convenience wrappers `submit_eval_midterm_to_ukr_rus_all_tasks.sh` and `submit_eval_ukr_rus_to_midterm_all_tasks.sh` just call the corresponding sbatch with the pre-existing model list files and accept the optional shots argument too:

```bash
# Override shots via the wrapper
bash scripts/submit_eval_midterm_to_ukr_rus_all_tasks.sh 1,5
```

---

## What the eval script does

For each checkpoint × task × shot count combination the script calls:

```bash
python3 experiments/run_single_experiment.py \
  --dataset <target_dataset> \
  --root <target_data_root> \
  --pretrained_model_run <ckpt_path> \
  --eval_only True \
  --eval_test_before_train True \
  --eval_val_before_train True \
  --save_roc_curve True \
  --task_name <neighbor_matching|temporal_link_prediction|classification> \
  --n_shots <shots> \
  --zero_shot <True if shots==0> \
  --prefix "trained_on_<model_name>_eval_on_<task_tag>_<shots>_shot" \
  ...  # task-specific args
```

Key points:
- `--eval_only True` freezes weights — no gradient updates happen on the target dataset.
- `--eval_test_before_train True` / `--eval_val_before_train True` run evaluation before any (hypothetical) training loop, so this is a pure zero/few-shot transfer.
- ROC curves are saved alongside the W&B logs.
- The prefix encodes the full provenance: which model was trained on what, evaluated on which task and how many shots.

---

## Adding a new source–target pair

1. Train on the new source dataset (or reuse an existing checkpoint).
2. Create a new model list file, e.g. `scripts/covid_train1_eval_on_midterm_model_list.txt`.
3. Submit directly against the target's sbatch script:
   ```bash
   sbatch scripts/eval_midterm_model_list_all_tasks.sbatch \
     scripts/covid_train1_eval_on_midterm_model_list.txt
   ```
   No new script is needed unless you want a convenience wrapper.
