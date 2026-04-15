# Experiment 1: Sequential Cross-Dataset Training (NM → eval all tasks)

## Hypothesis

Training on neighbor matching (NM) across two datasets and then evaluating on a held-out
dataset across all three tasks (NM, LP, PL) tests whether structural pretraining transfers
to unseen tasks — the core PRODIGY claim.

## Design

```
Train: midterm (NM) → fine-tune: covid19_twitter (NM) → eval: ukr_rus_twitter (NM + LP + PL)
```

- **Sequential order**: midterm first (larger, more diverse), covid second (closer domain
  to ukr_rus — both are non-US political Twitter retweet networks).
- **Training task**: neighbor matching only.
- **Eval tasks**: all three — NM (same task), LP and PL (cross-task transfer).
- **Eval shots**: 10-shot.

## Pipeline

### Step 1 — Pretrain on midterm NM

```bash
bash step1_submit_train_midterm.sh
```

Submits `train_single_task.sbatch` with `DATASET=midterm TASK=nm`. After the job
finishes, find the checkpoint:

```bash
ls state/train1_midterm_nm_*/state_dict
```

### Step 2 — Fine-tune on covid NM

```bash
bash step2_submit_finetune_covid.sh state/train1_midterm_nm_<run>/state_dict
```

Loads the midterm checkpoint and continues training on covid19_twitter NM. After the
job finishes, find the checkpoint:

```bash
ls state/train2_midterm_nm_to_covid_nm_*/state_dict
```

### Step 3 — Evaluate on ukr_rus (all tasks, 10-shot)

```bash
bash step3_submit_eval_ukr_rus.sh state/train2_midterm_nm_to_covid_nm_<run>/state_dict
```

Runs `eval_ukr_rus_twitter_model_list_all_tasks.sbatch` against the fine-tuned checkpoint,
evaluating NM, LP, and PL at 10 shots. Results are logged to W&B with prefixes of the form:

```
trained_on_midterm_nm_to_covid_nm_eval_on_{nm,lp,pl}_10_shot
```

## Baselines to compare against

| Model | Trained on | Eval on |
|-------|-----------|---------|
| train1_ukr_rus_nm | ukr_rus NM only | ukr_rus (NM, LP, PL) |
| train1_midterm_nm | midterm NM only | ukr_rus (NM, LP, PL) |
| **experiment_1** | midterm NM → covid NM | ukr_rus (NM, LP, PL) |

The single-dataset baselines should already exist from prior runs and can be pointed
at `eval_ukr_rus_twitter_model_list_all_tasks.sbatch` directly.
