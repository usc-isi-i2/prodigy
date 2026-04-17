# Experiment 7: midterm NM → covid LP → eval ukr_rus

## Design

```
Pretrain: midterm (NM) → Fine-tune: covid19_twitter (LP) → Eval: ukr_rus_twitter (NM + LP + PL)
```

Cross-task block. Held-out dataset: ukr_rus_twitter.

| | Exp 1 | Exp 4 | Exp 7 |
|-|-------|-------|-------|
| Pretrain | midterm NM | midterm LP | midterm NM |
| Fine-tune | covid NM | covid LP | covid LP |
| Eval | ukr_rus | ukr_rus | ukr_rus |

## Pipeline

```bash
# Step 1 — skipped, reuse midterm NM checkpoint from Experiment 1
# checkpoint: /home1/singhama/gfm/prodigy/state/train1_midterm_nm_15_04_2026_16_00_44/checkpoint/state_dict_8000.ckpt

# Step 2 — fine-tune on covid LP
bash step2_submit_finetune_covid.sh <midterm_nm_ckpt>
# → /home1/singhama/gfm/prodigy/state/exp7_train2_midterm_nm_to_covid_lp_*/state_dict

# Step 3 — eval on ukr_rus (NM + LP + PL, shots=1,5,10)
bash step3_submit_eval_ukr_rus.sh /home1/singhama/gfm/prodigy/state/exp7_train2_midterm_nm_to_covid_lp_<run>/state_dict
```

## Commands Run

```bash
# Step 2 — fine-tune on covid LP
bash scripts/cross_dataset/experiment_7/step2_submit_finetune_covid.sh \
  /home1/singhama/gfm/prodigy/state/train1_midterm_nm_15_04_2026_16_00_44/checkpoint/state_dict_8000.ckpt
```
