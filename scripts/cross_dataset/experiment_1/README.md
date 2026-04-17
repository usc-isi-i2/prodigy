# Experiment 1: midterm NM → covid NM → eval ukr_rus

## Design

```
Pretrain: midterm (NM) → Fine-tune: covid19_twitter (NM) → Eval: ukr_rus_twitter (NM + LP + PL)
```

Leave-one-out NM block. Held-out dataset: ukr_rus_twitter.

| | Exp 1 | Exp 2 | Exp 3 |
|-|-------|-------|-------|
| Pretrain | midterm NM | midterm NM | covid NM |
| Fine-tune | covid NM | ukr_rus NM | ukr_rus NM |
| Eval | ukr_rus | covid | midterm |

## Pipeline

```bash
# Step 1 — pretrain on midterm NM
bash step1_submit_train_midterm.sh
# → state/train1_midterm_nm_*/state_dict

# Step 2 — fine-tune on covid NM
bash step2_submit_finetune_covid.sh state/train1_midterm_nm_<run>/state_dict
# → state/train2_midterm_nm_to_covid_nm_*/state_dict

# Step 3 — eval on ukr_rus (NM + LP + PL, shots=1,5,10)
bash step3_submit_eval_ukr_rus.sh state/train2_midterm_nm_to_covid_nm_<run>/state_dict
```

## Commands Run

```bash
# Step 1 — pretrain on midterm NM
bash scripts/cross_dataset/experiment_1/step1_submit_train_midterm.sh
# checkpoint: state/train1_midterm_nm_15_04_2026_16_00_44/checkpoint/state_dict_8000.ckpt

# Step 2 — fine-tune on covid NM (job terminated early, used 8k-step checkpoint)
bash scripts/cross_dataset/experiment_1/step2_submit_finetune_covid.sh \
  state/train1_midterm_nm_15_04_2026_16_00_44/checkpoint/state_dict_8000.ckpt
# checkpoint: state/train2_midterm_nm_to_covid_nm_16_04_2026_10_07_00/state_dict

# Step 3 — eval on ukr_rus (NM + LP + PL, shots=1,5,10)
bash scripts/cross_dataset/experiment_1/step3_submit_eval_ukr_rus.sh \
  /home1/singhama/gfm/prodigy/state/train2_midterm_nm_to_covid_nm_16_04_2026_10_07_00/state_dict
```
