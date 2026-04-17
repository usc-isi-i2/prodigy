# Experiment 9: Cross-Task Sequential Training — covid NM → ukr_rus LP → eval midterm

## Design

```
Pretrain: covid19_twitter (NM) → Fine-tune: ukr_rus_twitter (LP) → Eval: midterm (NM + LP + PL)
```

Cross-task counterpart to Experiments 3 and 6. Tests whether switching task during fine-tuning
helps or hurts compared to staying on the same task.

| | Exp 3 | Exp 6 | **Exp 9** |
|-|-------|-------|-----------|
| Pretrain task | NM | LP | NM |
| Fine-tune task | NM | LP | **LP** |
| Train datasets | covid + ukr_rus | covid + ukr_rus | covid + ukr_rus |
| Eval dataset | midterm | midterm | midterm |

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

## Files

| File | Purpose |
|------|---------|
| `step2_finetune_ukr_rus.sbatch` | Fine-tune on ukr_rus_twitter LP (requires `CKPT_PATH`) |
| `step2_submit_finetune_ukr_rus.sh` | Submit step 2 with checkpoint path |
| `eval_midterm_model_list_all_tasks.sbatch` | Eval sbatch: runs NM + LP + PL on midterm |
| `step3_submit_eval_midterm.sh` | Build model list and submit eval job |

## Commands Run

```bash
# Step 2 — fine-tune on ukr_rus LP
bash scripts/cross_dataset/experiment_9/step2_submit_finetune_ukr_rus.sh \
  /home1/singhama/gfm/prodigy/state/exp3_train1_covid_nm_16_04_2026_10_36_14/state_dict
```
