# Experiment 6: covid LP → ukr_rus LP → eval midterm

## Overview

Sequential transfer learning using **link prediction (LP)** as the training task.

| Step | Dataset | Task | Epochs |
|------|---------|------|--------|
| Pretrain | covid19_twitter | temporal_link_prediction | 3 |
| Fine-tune | ukr_rus_twitter | temporal_link_prediction | 3 |
| Eval (held-out) | midterm | NM + LP + PL | shots: 1, 5, 10 |

## Rationale

Held-out dataset is midterm. We pretrain on covid (similar temporal retweet structure) then
fine-tune on ukr_rus (closer domain to midterm's political/election theme) before evaluating
zero/few-shot generalization on midterm across all three tasks.

## Files

| File | Purpose |
|------|---------|
| `step1_train_covid.sbatch` | Pretrain on covid19_twitter LP |
| `step1_submit_train_covid.sh` | Submit step 1 |
| `step2_finetune_ukr_rus.sbatch` | Fine-tune on ukr_rus_twitter LP (requires `CKPT_PATH`) |
| `step2_submit_finetune_ukr_rus.sh` | Submit step 2 with checkpoint path |
| `eval_midterm_model_list_all_tasks.sbatch` | Eval sbatch: runs NM + LP + PL on midterm |
| `step3_submit_eval_midterm.sh` | Build model list and submit eval job |

## Usage

```bash
# Step 1: Pretrain on covid LP
bash step1_submit_train_covid.sh

# After job completes, find the best checkpoint:
ls /home1/singhama/gfm/prodigy/state/exp6_train1_covid_lp_*/state_dict

# Step 2: Fine-tune on ukr_rus LP
bash step2_submit_finetune_ukr_rus.sh <step1_ckpt>

# After job completes, find the best checkpoint:
ls /home1/singhama/gfm/prodigy/state/exp6_train2_covid_lp_to_ukr_rus_lp_*/state_dict

# Step 3: Eval on midterm (NM + LP + PL, shots=1,5,10)
bash step3_submit_eval_midterm.sh <step2_ckpt>
```

## Key Args

### Step 1 (covid LP)
- `--midterm_edge_view temporal_history` — uses historical edges as context
- `--midterm_target_edge_view temporal_new` — predicts newly formed edges
- `--n_way 1 --n_shots 1 --n_query 3`

### Step 2 (ukr_rus LP fine-tune)
- Same LP config as step 1
- `--graph_filename retweet_graph_1p5m_hf03_political_labels.pt`

### Step 3 (midterm eval)
- LP uses `--midterm_target_edge_view default`
- NM uses `--n_way 3`, PL uses `--n_way 2`
- All tasks capped at `--val_len_cap 500 --test_len_cap 500`

## Logs

```
/home1/singhama/gfm/prodigy/logs/exp6_train1_covid_lp_<job_id>.out
/home1/singhama/gfm/prodigy/logs/exp6_train2_covid_lp_to_ukr_rus_lp_<job_id>.out
/home1/singhama/gfm/prodigy/logs/eval_midterm_all_<job_id>.out
```

Results are logged to W&B project `graph-clip`.

## Commands Run

```bash
# Step 1 — pretrain on covid LP
bash scripts/cross_dataset/experiment_6/step1_submit_train_covid.sh
# checkpoint: state/exp6_train1_covid_lp_16_04_2026_16_23_18/state_dict

# Step 2 — fine-tune on ukr_rus LP
bash scripts/cross_dataset/experiment_6/step2_submit_finetune_ukr_rus.sh \
  /home1/singhama/gfm/prodigy/state/exp6_train1_covid_lp_16_04_2026_16_23_18/state_dict
# checkpoint: state/exp6_train2_covid_lp_to_ukr_rus_lp_16_04_2026_17_24_43/state_dict

# Step 3 — eval on midterm (NM + LP + PL, shots=1,5,10)
bash scripts/cross_dataset/experiment_6/step3_submit_eval_midterm.sh \
  /home1/singhama/gfm/prodigy/state/exp6_train2_covid_lp_to_ukr_rus_lp_16_04_2026_17_24_43/state_dict
```
