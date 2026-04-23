# Experiment 10: midterm LP → covid NM → eval ukr_rus

## Design

```
Pretrain: midterm (LP) → Fine-tune: covid19_twitter (NM) → Eval: ukr_rus_twitter (NM + LP + PL)
```

Cross-task block (LP→NM). Held-out dataset: ukr_rus_twitter.

| | Exp 4 | Exp 10 |
|-|-------|--------|
| Pretrain | midterm LP | midterm LP |
| Fine-tune | covid LP | covid NM |
| Eval | ukr_rus | ukr_rus |

## Pipeline

```bash
# Step 1 — skipped, reuse midterm LP checkpoint from Experiment 4
# checkpoint: /home1/singhama/gfm/prodigy/state/exp4_train1_midterm_lp_16_04_2026_16_23_15/state_dict

# Step 2 — fine-tune on covid NM
bash step2_submit_finetune_covid.sh <midterm_lp_ckpt>
# → /home1/singhama/gfm/prodigy/state/exp10_train2_midterm_lp_to_covid_nm_22_04_2026_13_24_06/state_dict

# Step 3 — eval on ukr_rus (NM + LP + PL, shots=1,5,10)
bash step3_submit_eval_ukr_rus.sh /home1/singhama/gfm/prodigy/state/exp10_train2_midterm_lp_to_covid_nm_<run>/state_dict
```

## Commands Run

```bash
# Step 2 — fine-tune on covid NM
bash scripts/cross_dataset/experiment_10/step2_submit_finetune_covid.sh \
  /home1/singhama/gfm/prodigy/state/exp4_train1_midterm_lp_16_04_2026_16_23_15/state_dict

# Step 3 — eval on ukr_rus (NM + LP + PL, shots=1,5,10)
bash scripts/cross_dataset/experiment_10/step3_submit_eval_ukr_rus.sh \
  /home1/singhama/gfm/prodigy/state/exp10_train2_midterm_lp_to_covid_nm_22_04_2026_13_24_06/state_dict
```
