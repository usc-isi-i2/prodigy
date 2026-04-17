# Experiment 4: midterm LP → covid LP → eval ukr_rus

## Design

```
Pretrain: midterm (LP) → Fine-tune: covid19_twitter (LP) → Eval: ukr_rus_twitter (NM + LP + PL)
```

Leave-one-out LP block. Held-out dataset: ukr_rus_twitter.

| | Exp 4 | Exp 5 | Exp 6 |
|-|-------|-------|-------|
| Pretrain | midterm LP | midterm LP | covid LP |
| Fine-tune | covid LP | ukr_rus LP | ukr_rus LP |
| Eval | ukr_rus | covid | midterm |

## Pipeline

```bash
# Step 1 — pretrain on midterm LP
bash step1_submit_train_midterm.sh
# → state/exp4_train1_midterm_lp_*/state_dict

# Step 2 — fine-tune on covid LP
bash step2_submit_finetune_covid.sh state/exp4_train1_midterm_lp_<run>/state_dict
# → state/exp4_train2_midterm_lp_to_covid_lp_*/state_dict

# Step 3 — eval on ukr_rus (NM + LP + PL, shots=1,5,10)
bash step3_submit_eval_ukr_rus.sh state/exp4_train2_midterm_lp_to_covid_lp_<run>/state_dict
```

## Commands Run

```bash
# Step 1 — pretrain on midterm LP
bash scripts/cross_dataset/experiment_4/step1_submit_train_midterm.sh
# checkpoint: state/exp4_train1_midterm_lp_16_04_2026_16_23_15/state_dict

# Step 2 — fine-tune on covid LP
bash scripts/cross_dataset/experiment_4/step2_submit_finetune_covid.sh \
  /home1/singhama/gfm/prodigy/state/exp4_train1_midterm_lp_16_04_2026_16_23_15/state_dict
# checkpoint: state/exp4_train2_midterm_lp_to_covid_lp_16_04_2026_17_31_13/state_dict

# Step 3 — eval on ukr_rus (NM + LP + PL, shots=1,5,10)
bash scripts/cross_dataset/experiment_4/step3_submit_eval_ukr_rus.sh \
  /home1/singhama/gfm/prodigy/state/exp4_train2_midterm_lp_to_covid_lp_16_04_2026_17_31_13/state_dict
```
