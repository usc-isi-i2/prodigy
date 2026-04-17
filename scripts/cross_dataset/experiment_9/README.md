# Experiment 9: covid NM → ukr_rus LP → eval midterm

## Design

```
Pretrain: covid19_twitter (NM) → Fine-tune: ukr_rus_twitter (LP) → Eval: midterm (NM + LP + PL)
```

Cross-task block. Held-out dataset: midterm.

| | Exp 3 | Exp 6 | Exp 9 |
|-|-------|-------|-------|
| Pretrain | covid NM | covid LP | covid NM |
| Fine-tune | ukr_rus NM | ukr_rus LP | ukr_rus LP |
| Eval | midterm | midterm | midterm |

## Pipeline

```bash
# Step 1 — skipped, reuse covid NM checkpoint from Experiment 3
# checkpoint: state/exp3_train1_covid_nm_16_04_2026_10_36_14/state_dict

# Step 2 — fine-tune on ukr_rus LP
bash step2_submit_finetune_ukr_rus.sh <covid_nm_ckpt>
# → state/exp9_train2_covid_nm_to_ukr_rus_lp_*/state_dict

# Step 3 — eval on midterm (NM + LP + PL, shots=1,5,10)
bash step3_submit_eval_midterm.sh state/exp9_train2_covid_nm_to_ukr_rus_lp_<run>/state_dict
```

## Commands Run

```bash
# Step 2 — fine-tune on ukr_rus LP
bash scripts/cross_dataset/experiment_9/step2_submit_finetune_ukr_rus.sh \
  /home1/singhama/gfm/prodigy/state/exp3_train1_covid_nm_16_04_2026_10_36_14/state_dict
```
