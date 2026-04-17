# Experiment 8: Cross-Task Sequential Training — midterm NM → ukr_rus LP → eval covid

## Design

```
Pretrain: midterm (NM) → Fine-tune: ukr_rus_twitter (LP) → Eval: covid19_twitter (NM + LP)
```

Cross-task counterpart to Experiments 2 and 5. Tests whether switching task during fine-tuning
helps or hurts compared to staying on the same task.

| | Exp 2 | Exp 5 | **Exp 8** |
|-|-------|-------|-----------|
| Pretrain task | NM | LP | NM |
| Fine-tune task | NM | LP | **LP** |
| Train datasets | midterm + ukr_rus | midterm + ukr_rus | midterm + ukr_rus |
| Eval dataset | covid | covid | covid |

## Pipeline

```bash
# Step 1 — skipped, reuse midterm NM checkpoint from Experiment 1
# checkpoint: state/train1_midterm_nm_15_04_2026_16_00_44/checkpoint/state_dict_8000.ckpt

# Step 2 — fine-tune on ukr_rus LP
bash step2_submit_finetune_ukr_rus.sh <midterm_nm_ckpt>
# → state/exp8_train2_midterm_nm_to_ukr_rus_lp_*/state_dict

# Step 3 — eval on covid (NM + LP, shots=1,5,10)
bash step3_submit_eval_covid.sh state/exp8_train2_midterm_nm_to_ukr_rus_lp_<run>/state_dict
```

## Files

| File | Purpose |
|------|---------|
| `step2_finetune_ukr_rus.sbatch` | Fine-tune on ukr_rus_twitter LP (requires `CKPT_PATH`) |
| `step2_submit_finetune_ukr_rus.sh` | Submit step 2 with checkpoint path |
| `eval_covid19_twitter_model_list_all_tasks.sbatch` | Eval sbatch: runs NM + LP on covid |
| `step3_submit_eval_covid.sh` | Build model list and submit eval job |

## Commands Run

```bash
# Step 2 — fine-tune on ukr_rus LP
bash scripts/cross_dataset/experiment_8/step2_submit_finetune_ukr_rus.sh \
  state/train1_midterm_nm_15_04_2026_16_00_44/checkpoint/state_dict_8000.ckpt
```
