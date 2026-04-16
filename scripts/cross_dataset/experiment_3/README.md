# Experiment 3: Sequential Cross-Dataset Training — Eval on midterm

## Design

```
Train: covid19_twitter (NM) → fine-tune: ukr_rus_twitter (NM) → eval: midterm (NM + LP + PL)
```

Part of a leave-one-out sweep across all three datasets:
- Experiment 1: train midterm+covid, eval ukr_rus
- Experiment 2: train midterm+ukr_rus, eval covid
- **Experiment 3: train covid+ukr_rus, eval midterm**

## Pipeline

```bash
# Step 1 — pretrain on covid NM
bash step1_submit_train_covid.sh
# → state/exp3_train1_covid_nm_*/checkpoint/

# Step 2 — fine-tune on ukr_rus NM
bash step2_submit_finetune_ukr_rus.sh state/exp3_train1_covid_nm_<run>/checkpoint/state_dict_<best_step>.ckpt
# → state/exp3_train2_covid_nm_to_ukr_rus_nm_*/checkpoint/

# Step 3 — eval on midterm (NM + LP + PL, shots=1,5,10)
bash step3_submit_eval_midterm.sh state/exp3_train2_covid_nm_to_ukr_rus_nm_<run>/checkpoint/state_dict_<best_step>.ckpt
```

## Commands Run

```bash
# Step 1 — pretrain on covid NM
bash scripts/cross_dataset/experiment_3/step1_submit_train_covid.sh
# checkpoint: state/exp3_train1_covid_nm_16_04_2026_10_36_14/state_dict

# Step 2 — fine-tune on ukr_rus NM
bash scripts/cross_dataset/experiment_3/step2_submit_finetune_ukr_rus.sh \
  /home1/singhama/gfm/prodigy/state/exp3_train1_covid_nm_16_04_2026_10_36_14/state_dict
```
