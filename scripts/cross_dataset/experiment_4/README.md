# Experiment 4: Sequential LP Training — Eval on ukr_rus_twitter

## Design

```
Train: midterm (LP) → fine-tune: covid19_twitter (LP) → eval: ukr_rus_twitter (NM + LP + PL)
```

LP counterpart to Experiment 1 (NM). Tests whether LP pretraining transfers to NM and PL tasks.

| | Exp 1 | Exp 4 |
|-|-------|-------|
| Train task | NM | LP |
| Train datasets | midterm + covid | midterm + covid |
| Eval dataset | ukr_rus | ukr_rus |

## Pipeline

```bash
# Step 1 — pretrain on midterm LP
bash step1_submit_train_midterm.sh
# → state/exp4_train1_midterm_lp_*/checkpoint/

# Step 2 — fine-tune on covid LP
bash step2_submit_finetune_covid.sh state/exp4_train1_midterm_lp_<run>/checkpoint/state_dict_<best>.ckpt
# → state/exp4_train2_midterm_lp_to_covid_lp_*/checkpoint/

# Step 3 — eval on ukr_rus (NM + LP + PL, 1,5,10-shot)
bash step3_submit_eval_ukr_rus.sh state/exp4_train2_midterm_lp_to_covid_lp_<run>/checkpoint/state_dict_<best>.ckpt
```
