# Experiment 11: midterm LP → ukr_rus NM → eval covid

## Design

```
Pretrain: midterm (LP) → Fine-tune: ukr_rus_twitter (NM) → Eval: covid19_twitter (NM + LP + PL)
```

Cross-task block (LP→NM). Held-out dataset: covid19_twitter.

| | Exp 5 | Exp 11 |
|-|-------|--------|
| Pretrain | midterm LP | midterm LP |
| Fine-tune | ukr_rus LP | ukr_rus NM |
| Eval | covid | covid |

## Pipeline

```bash
# Step 1 — skipped, reuse midterm LP checkpoint from Experiment 4
# checkpoint: /home1/singhama/gfm/prodigy/state/exp4_train1_midterm_lp_16_04_2026_16_23_15/state_dict

# Step 2 — fine-tune on ukr_rus NM
bash step2_submit_finetune_ukr_rus.sh <midterm_lp_ckpt>
# → /home1/singhama/gfm/prodigy/state/exp11_train2_midterm_lp_to_ukr_rus_nm_*/state_dict

# Step 3 — eval on covid (NM + LP + PL, shots=1,5,10)
bash step3_submit_eval_covid.sh /home1/singhama/gfm/prodigy/state/exp11_train2_midterm_lp_to_ukr_rus_nm_<run>/state_dict
```

## Commands Run

```bash
# Step 2 — fine-tune on ukr_rus NM
bash scripts/cross_dataset/experiment_11/step2_submit_finetune_ukr_rus.sh \
  /home1/singhama/gfm/prodigy/state/exp4_train1_midterm_lp_16_04_2026_16_23_15/state_dict

# Step 3 — eval on covid (NM + LP + PL, shots=1,5,10)
bash scripts/cross_dataset/experiment_11/step3_submit_eval_covid.sh \
  /home1/singhama/gfm/prodigy/state/exp11_train2_midterm_lp_to_ukr_rus_nm_<run>/state_dict
```
