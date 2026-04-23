# Experiment 13: covid NM → midterm NM → eval ukr_rus

## Design

```
Pretrain: covid19_twitter (NM) → Fine-tune: midterm (NM) → Eval: ukr_rus_twitter (NM + LP + PL)
```

Dataset-order reversal of Experiment 1. Held-out dataset: ukr_rus_twitter.

| | Exp 1 | Exp 13 |
|-|-------|--------|
| Pretrain | midterm NM | covid NM |
| Fine-tune | covid NM | midterm NM |
| Eval | ukr_rus | ukr_rus |

## Pipeline

```bash
# Step 1 — skipped, reuse covid NM checkpoint from Experiment 3
# checkpoint: /home1/singhama/gfm/prodigy/state/exp3_train1_covid_nm_16_04_2026_10_36_14/state_dict

# Step 2 — fine-tune on midterm NM
bash step2_submit_finetune_midterm.sh <covid_nm_ckpt>
# → /home1/singhama/gfm/prodigy/state/exp13_train2_covid_nm_to_midterm_nm_*/state_dict

# Step 3 — eval on ukr_rus (NM + LP + PL, shots=1,5,10)
bash step3_submit_eval_ukr_rus.sh /home1/singhama/gfm/prodigy/state/exp13_train2_covid_nm_to_midterm_nm_<run>/state_dict
```

## Commands Run

```bash
# Step 2 — fine-tune on midterm NM
bash scripts/cross_dataset/experiment_13/step2_submit_finetune_midterm.sh \
  /home1/singhama/gfm/prodigy/state/exp3_train1_covid_nm_16_04_2026_10_36_14/state_dict

# Step 3 — eval on ukr_rus (NM + LP + PL, shots=1,5,10)
bash scripts/cross_dataset/experiment_13/step3_submit_eval_ukr_rus.sh \
  /home1/singhama/gfm/prodigy/state/exp13_train2_covid_nm_to_midterm_nm_<run>/state_dict
```
