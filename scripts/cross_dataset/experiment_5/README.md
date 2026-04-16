# Experiment 5: Sequential LP Training — Eval on covid19_twitter

## Design

```
Train: midterm (LP) → fine-tune: ukr_rus_twitter (LP) → eval: covid19_twitter (NM + LP)
```

LP counterpart to Experiment 2 (NM).

| | Exp 2 | Exp 5 |
|-|-------|-------|
| Train task | NM | LP |
| Train datasets | midterm + ukr_rus | midterm + ukr_rus |
| Eval dataset | covid | covid |

## Pipeline

```bash
# Step 1 — pretrain on midterm LP (or reuse exp4 checkpoint)
bash step1_submit_train_midterm.sh
# → state/exp5_train1_midterm_lp_*/checkpoint/

# Step 2 — fine-tune on ukr_rus LP
bash step2_submit_finetune_ukr_rus.sh state/exp5_train1_midterm_lp_<run>/checkpoint/state_dict_<best>.ckpt
# → state/exp5_train2_midterm_lp_to_ukr_rus_lp_*/checkpoint/

# Step 3 — eval on covid (NM + LP, 1,5,10-shot)
bash step3_submit_eval_covid.sh state/exp5_train2_midterm_lp_to_ukr_rus_lp_<run>/checkpoint/state_dict_<best>.ckpt
```
