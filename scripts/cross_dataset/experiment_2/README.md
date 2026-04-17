# Experiment 2: midterm NM → ukr_rus NM → eval covid

## Design

```
Pretrain: midterm (NM) → Fine-tune: ukr_rus_twitter (NM) → Eval: covid19_twitter (NM + LP)
```

Leave-one-out NM block. Held-out dataset: covid19_twitter (no PL task).

| | Exp 1 | Exp 2 | Exp 3 |
|-|-------|-------|-------|
| Pretrain | midterm NM | midterm NM | covid NM |
| Fine-tune | covid NM | ukr_rus NM | ukr_rus NM |
| Eval | ukr_rus | covid | midterm |

## Pipeline

```bash
# Step 1 — pretrain on midterm NM
bash step1_submit_train_midterm.sh
# → /home1/singhama/gfm/prodigy/state/exp2_train1_midterm_nm_*/state_dict

# Step 2 — fine-tune on ukr_rus NM
bash step2_submit_finetune_ukr_rus.sh /home1/singhama/gfm/prodigy/state/exp2_train1_midterm_nm_<run>/state_dict
# → /home1/singhama/gfm/prodigy/state/exp2_train2_midterm_nm_to_ukr_rus_nm_*/state_dict

# Step 3 — eval on covid (NM + LP, shots=1,5,10)
bash step3_submit_eval_covid.sh /home1/singhama/gfm/prodigy/state/exp2_train2_midterm_nm_to_ukr_rus_nm_<run>/state_dict
```

## Commands Run

```bash
# Step 1 — skipped, reused midterm NM checkpoint from Experiment 1
# checkpoint: /home1/singhama/gfm/prodigy/state/train1_midterm_nm_15_04_2026_16_00_44/checkpoint/state_dict_8000.ckpt

# Step 2 — fine-tune on ukr_rus NM
bash scripts/cross_dataset/experiment_2/step2_submit_finetune_ukr_rus.sh \
  /home1/singhama/gfm/prodigy/state/train1_midterm_nm_15_04_2026_16_00_44/checkpoint/state_dict_8000.ckpt
# checkpoint: /home1/singhama/gfm/prodigy/state/exp2_train2_midterm_nm_to_ukr_rus_nm_16_04_2026_10_32_38/checkpoint/state_dict_27000.ckpt

# Step 3 — eval on covid (NM + LP, shots=1,5,10)
bash scripts/cross_dataset/experiment_2/step3_submit_eval_covid.sh \
  /home1/singhama/gfm/prodigy/state/exp2_train2_midterm_nm_to_ukr_rus_nm_16_04_2026_10_32_38/checkpoint/state_dict_27000.ckpt
```
