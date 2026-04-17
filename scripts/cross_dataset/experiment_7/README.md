# Experiment 7: Cross-Task Sequential Training — midterm NM → covid LP → eval ukr_rus

## Design

```
Pretrain: midterm (NM) → Fine-tune: covid19_twitter (LP) → Eval: ukr_rus_twitter (NM + LP + PL)
```

Cross-task counterpart to Experiments 1 and 4. Tests whether switching task during fine-tuning
helps or hurts compared to staying on the same task.

| | Exp 1 | Exp 4 | **Exp 7** |
|-|-------|-------|-----------|
| Pretrain task | NM | LP | NM |
| Fine-tune task | NM | LP | **LP** |
| Train datasets | midterm + covid | midterm + covid | midterm + covid |
| Eval dataset | ukr_rus | ukr_rus | ukr_rus |

## Pipeline

```bash
# Step 1 — skipped, reuse midterm NM checkpoint from Experiment 1
# checkpoint: state/train1_midterm_nm_15_04_2026_16_00_44/checkpoint/state_dict_8000.ckpt

# Step 2 — fine-tune on covid LP
bash step2_submit_finetune_covid.sh <midterm_nm_ckpt>
# → state/exp7_train2_midterm_nm_to_covid_lp_*/state_dict

# Step 3 — eval on ukr_rus (NM + LP + PL, shots=1,5,10)
bash step3_submit_eval_ukr_rus.sh state/exp7_train2_midterm_nm_to_covid_lp_<run>/state_dict
```

## Files

| File | Purpose |
|------|---------|
| `step2_finetune_covid.sbatch` | Fine-tune on covid19_twitter LP (requires `CKPT_PATH`) |
| `step2_submit_finetune_covid.sh` | Submit step 2 with checkpoint path |
| `eval_ukr_rus_twitter_model_list_all_tasks.sbatch` | Eval sbatch: runs NM + LP + PL on ukr_rus |
| `step3_submit_eval_ukr_rus.sh` | Build model list and submit eval job |

## Commands Run

```bash
# Step 2 — fine-tune on covid LP
bash scripts/cross_dataset/experiment_7/step2_submit_finetune_covid.sh \
  state/train1_midterm_nm_15_04_2026_16_00_44/checkpoint/state_dict_8000.ckpt
```
