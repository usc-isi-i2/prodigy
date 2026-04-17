# Experiment 5: midterm LP → ukr_rus LP → eval covid

## Design

```
Pretrain: midterm (LP) → Fine-tune: ukr_rus_twitter (LP) → Eval: covid19_twitter (NM + LP)
```

Leave-one-out LP block. Held-out dataset: covid19_twitter (no PL task).

| | Exp 4 | Exp 5 | Exp 6 |
|-|-------|-------|-------|
| Pretrain | midterm LP | midterm LP | covid LP |
| Fine-tune | covid LP | ukr_rus LP | ukr_rus LP |
| Eval | ukr_rus | covid | midterm |

## Pipeline

```bash
# Step 1 — pretrain on midterm LP
bash step1_submit_train_midterm.sh
# → /home1/singhama/gfm/prodigy/state/exp5_train1_midterm_lp_*/state_dict

# Step 2 — fine-tune on ukr_rus LP
bash step2_submit_finetune_ukr_rus.sh /home1/singhama/gfm/prodigy/state/exp5_train1_midterm_lp_<run>/state_dict
# → /home1/singhama/gfm/prodigy/state/exp5_train2_midterm_lp_to_ukr_rus_lp_*/state_dict

# Step 3 — eval on covid (NM + LP, shots=1,5,10)
bash step3_submit_eval_covid.sh /home1/singhama/gfm/prodigy/state/exp5_train2_midterm_lp_to_ukr_rus_lp_<run>/state_dict
```

## Commands Run

```bash
# Step 1 — skipped, reused midterm LP checkpoint from Experiment 4
# checkpoint: /home1/singhama/gfm/prodigy/state/exp4_train1_midterm_lp_16_04_2026_16_23_15/state_dict

# Step 2 — fine-tune on ukr_rus LP
bash scripts/cross_dataset/experiment_5/step2_submit_finetune_ukr_rus.sh \
  /home1/singhama/gfm/prodigy/state/exp4_train1_midterm_lp_16_04_2026_16_23_15/state_dict
# checkpoint: /home1/singhama/gfm/prodigy/state/exp5_train2_midterm_lp_to_ukr_rus_lp_16_04_2026_17_48_24/state_dict

# Step 3 — eval on covid (NM + LP, shots=1,5,10)
bash scripts/cross_dataset/experiment_5/step3_submit_eval_covid.sh \
  /home1/singhama/gfm/prodigy/state/exp5_train2_midterm_lp_to_ukr_rus_lp_16_04_2026_17_48_24/state_dict
```
