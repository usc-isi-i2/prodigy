# Experiment 3: covid NM → ukr_rus NM → eval midterm

## Design

```
Pretrain: covid19_twitter (NM) → Fine-tune: ukr_rus_twitter (NM) → Eval: midterm (NM + LP + PL)
```

Leave-one-out NM block. Held-out dataset: midterm.

| | Exp 1 | Exp 2 | Exp 3 |
|-|-------|-------|-------|
| Pretrain | midterm NM | midterm NM | covid NM |
| Fine-tune | covid NM | ukr_rus NM | ukr_rus NM |
| Eval | ukr_rus | covid | midterm |

## Pipeline

```bash
# Step 1 — pretrain on covid NM
bash step1_submit_train_covid.sh
# → /home1/singhama/gfm/prodigy/state/exp3_train1_covid_nm_*/state_dict

# Step 2 — fine-tune on ukr_rus NM
bash step2_submit_finetune_ukr_rus.sh /home1/singhama/gfm/prodigy/state/exp3_train1_covid_nm_<run>/state_dict
# → /home1/singhama/gfm/prodigy/state/exp3_train2_covid_nm_to_ukr_rus_nm_*/state_dict

# Step 3 — eval on midterm (NM + LP + PL, shots=1,5,10)
bash step3_submit_eval_midterm.sh /home1/singhama/gfm/prodigy/state/exp3_train2_covid_nm_to_ukr_rus_nm_<run>/state_dict
```

## Commands Run

```bash
# Step 1 — pretrain on covid NM
bash scripts/cross_dataset/experiment_3/step1_submit_train_covid.sh
# checkpoint: /home1/singhama/gfm/prodigy/state/exp3_train1_covid_nm_16_04_2026_10_36_14/state_dict

# Step 2 — fine-tune on ukr_rus NM
bash scripts/cross_dataset/experiment_3/step2_submit_finetune_ukr_rus.sh \
  /home1/singhama/gfm/prodigy/state/exp3_train1_covid_nm_16_04_2026_10_36_14/state_dict
# checkpoint: /home1/singhama/gfm/prodigy/state/exp3_train2_covid_nm_to_ukr_rus_nm_16_04_2026_13_39_26/checkpoint/state_dict_30000.ckpt

# Step 3 — eval on midterm (NM + LP + PL, shots=1,5,10)
bash scripts/cross_dataset/experiment_3/step3_submit_eval_midterm.sh \
  /home1/singhama/gfm/prodigy/state/exp3_train2_covid_nm_to_ukr_rus_nm_16_04_2026_13_39_26/checkpoint/state_dict_30000.ckpt
```
