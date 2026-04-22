# Experiment 12: covid LP → ukr_rus NM → eval midterm

## Design

```
Pretrain: covid19_twitter (LP) → Fine-tune: ukr_rus_twitter (NM) → Eval: midterm (NM + LP + PL)
```

Cross-task block (LP→NM). Held-out dataset: midterm.

| | Exp 6 | Exp 12 |
|-|-------|--------|
| Pretrain | covid LP | covid LP |
| Fine-tune | ukr_rus LP | ukr_rus NM |
| Eval | midterm | midterm |

## Pipeline

```bash
# Step 1 — skipped, reuse covid LP checkpoint from Experiment 6
# checkpoint: /home1/singhama/gfm/prodigy/state/exp6_train1_covid_lp_16_04_2026_16_23_18/state_dict

# Step 2 — fine-tune on ukr_rus NM
bash step2_submit_finetune_ukr_rus.sh <covid_lp_ckpt>
# → /home1/singhama/gfm/prodigy/state/exp12_train2_covid_lp_to_ukr_rus_nm_*/state_dict

# Step 3 — eval on midterm (NM + LP + PL, shots=1,5,10)
bash step3_submit_eval_midterm.sh /home1/singhama/gfm/prodigy/state/exp12_train2_covid_lp_to_ukr_rus_nm_<run>/state_dict
```

## Commands Run

```bash
# Step 2 — fine-tune on ukr_rus NM
bash scripts/cross_dataset/experiment_12/step2_submit_finetune_ukr_rus.sh \
  /home1/singhama/gfm/prodigy/state/exp6_train1_covid_lp_16_04_2026_16_23_18/state_dict

# Step 3 — eval on midterm (NM + LP + PL, shots=1,5,10)
bash scripts/cross_dataset/experiment_12/step3_submit_eval_midterm.sh \
  /home1/singhama/gfm/prodigy/state/exp12_train2_covid_lp_to_ukr_rus_nm_<run>/state_dict
```
