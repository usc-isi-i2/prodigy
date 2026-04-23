# Experiment 15: ukr_rus NM → covid NM → eval midterm

## Design

```
Pretrain: ukr_rus_twitter (NM) → Fine-tune: covid19_twitter (NM) → Eval: midterm (NM + LP + PL)
```

Dataset-order reversal of Experiment 3. Held-out dataset: midterm.

| | Exp 3 | Exp 15 |
|-|-------|--------|
| Pretrain | covid NM | ukr_rus NM |
| Fine-tune | ukr_rus NM | covid NM |
| Eval | midterm | midterm |

## Pipeline

```bash
# Step 1 — skipped, reuse ukr_rus NM checkpoint from Experiment 14
# checkpoint: /home1/singhama/gfm/prodigy/state/exp14_train1_ukr_rus_nm_<run>/state_dict

# Step 2 — fine-tune on covid NM
bash step2_submit_finetune_covid.sh <ukr_rus_nm_ckpt>
# → /home1/singhama/gfm/prodigy/state/exp15_train2_ukr_rus_nm_to_covid_nm_*/state_dict

# Step 3 — eval on midterm (NM + LP + PL, shots=1,5,10)
bash step3_submit_eval_midterm.sh /home1/singhama/gfm/prodigy/state/exp15_train2_ukr_rus_nm_to_covid_nm_<run>/state_dict
```

## Commands Run

```bash
# Step 2 — fine-tune on covid NM
bash scripts/cross_dataset/experiment_15/step2_submit_finetune_covid.sh \
  /home1/singhama/gfm/prodigy/state/exp14_train1_ukr_rus_nm_<run>/state_dict

# Step 3 — eval on midterm (NM + LP + PL, shots=1,5,10)
bash scripts/cross_dataset/experiment_15/step3_submit_eval_midterm.sh \
  /home1/singhama/gfm/prodigy/state/exp15_train2_ukr_rus_nm_to_covid_nm_<run>/state_dict
```
