# Experiment 6: covid LP → ukr_rus LP → eval midterm

## Design

```
Pretrain: covid19_twitter (LP) → Fine-tune: ukr_rus_twitter (LP) → Eval: midterm (NM + LP + PL)
```

Leave-one-out LP block. Held-out dataset: midterm.

| | Exp 4 | Exp 5 | Exp 6 |
|-|-------|-------|-------|
| Pretrain | midterm LP | midterm LP | covid LP |
| Fine-tune | covid LP | ukr_rus LP | ukr_rus LP |
| Eval | ukr_rus | covid | midterm |

## Pipeline

```bash
# Step 1 — pretrain on covid LP
bash step1_submit_train_covid.sh
# → state/exp6_train1_covid_lp_*/state_dict

# Step 2 — fine-tune on ukr_rus LP
bash step2_submit_finetune_ukr_rus.sh state/exp6_train1_covid_lp_<run>/state_dict
# → state/exp6_train2_covid_lp_to_ukr_rus_lp_*/state_dict

# Step 3 — eval on midterm (NM + LP + PL, shots=1,5,10)
bash step3_submit_eval_midterm.sh state/exp6_train2_covid_lp_to_ukr_rus_lp_<run>/state_dict
```

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
