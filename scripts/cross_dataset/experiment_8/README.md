# Experiment 8: midterm NM → ukr_rus LP → eval covid

## Design

```
Pretrain: midterm (NM) → Fine-tune: ukr_rus_twitter (LP) → Eval: covid19_twitter (NM + LP + PL)
```

Cross-task block. Held-out dataset: covid19_twitter.

| | Exp 2 | Exp 5 | Exp 8 |
|-|-------|-------|-------|
| Pretrain | midterm NM | midterm LP | midterm NM |
| Fine-tune | ukr_rus NM | ukr_rus LP | ukr_rus LP |
| Eval | covid | covid | covid |

## Pipeline

```bash
# Step 1 — skipped, reuse midterm NM checkpoint from Experiment 1
# checkpoint: /home1/singhama/gfm/prodigy/state/train1_midterm_nm_15_04_2026_16_00_44/checkpoint/state_dict_8000.ckpt

# Step 2 — fine-tune on ukr_rus LP
bash step2_submit_finetune_ukr_rus.sh <midterm_nm_ckpt>
# → /home1/singhama/gfm/prodigy/state/exp8_train2_midterm_nm_to_ukr_rus_lp_*/state_dict

# Step 3 — eval on covid (NM + LP + PL, shots=1,5,10)
bash step3_submit_eval_covid.sh /home1/singhama/gfm/prodigy/state/exp8_train2_midterm_nm_to_ukr_rus_lp_<run>/state_dict
```

## Commands Run

```bash
# Step 2 — fine-tune on ukr_rus LP
bash scripts/cross_dataset/experiment_8/step2_submit_finetune_ukr_rus.sh \
  /home1/singhama/gfm/prodigy/state/train1_midterm_nm_15_04_2026_16_00_44/checkpoint/state_dict_8000.ckpt
```
