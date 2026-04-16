# Experiment 2: Sequential Cross-Dataset Training — Eval on covid19_twitter

## Design

```
Train: midterm (NM) → fine-tune: ukr_rus_twitter (NM) → eval: covid19_twitter (NM + LP)
```

Part of a leave-one-out sweep across all three datasets:
- Experiment 1: train midterm+covid, eval ukr_rus
- **Experiment 2: train midterm+ukr_rus, eval covid**
- Experiment 3: train covid+ukr_rus, eval midterm

## Pipeline

```bash
# Step 1 — pretrain on midterm NM
bash step1_submit_train_midterm.sh
# → state/exp2_train1_midterm_nm_*/state_dict

# Step 2 — fine-tune on ukr_rus NM
bash step2_submit_finetune_ukr_rus.sh state/exp2_train1_midterm_nm_<run>/checkpoint/state_dict_<best_step>.ckpt
# → state/exp2_train2_midterm_nm_to_ukr_rus_nm_*/state_dict

# Step 3 — eval on covid (NM + LP, 10-shot)
bash step3_submit_eval_covid.sh state/exp2_train2_midterm_nm_to_ukr_rus_nm_<run>/checkpoint/state_dict_<best_step>.ckpt
```
