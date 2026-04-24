# Experiment 14: ukr_rus NM → midterm NM → eval covid

## Design

```
Pretrain: ukr_rus_twitter (NM) → Fine-tune: midterm (NM) → Eval: covid19_twitter (NM + LP + PL)
```

Dataset-order reversal of Experiment 2. Held-out dataset: covid19_twitter.

| | Exp 2 | Exp 14 |
|-|-------|--------|
| Pretrain | midterm NM | ukr_rus NM |
| Fine-tune | ukr_rus NM | midterm NM |
| Eval | covid | covid |

## Pipeline

```bash
# Step 1 — pretrain on ukr_rus NM
bash step1_submit_train_ukr_rus.sh
# → /home1/singhama/gfm/prodigy/state/exp14_train1_ukr_rus_nm_23_04_2026_11_01_22/state_dict
# Note: this checkpoint is also reused by Experiment 15

# Step 2 — fine-tune on midterm NM
bash step2_submit_finetune_midterm.sh <ukr_rus_nm_ckpt>
# → /home1/singhama/gfm/prodigy/state/exp14_train2_ukr_rus_nm_to_midterm_nm_23_04_2026_13_26_26/checkpoint/state_dict_12000.ckpt
# Note: job hit time limit; using intermediate checkpoint state_dict_12000.ckpt

# Step 3 — eval on covid (NM + LP + PL, shots=1,5,10)
bash step3_submit_eval_covid.sh /home1/singhama/gfm/prodigy/state/exp14_train2_ukr_rus_nm_to_midterm_nm_23_04_2026_13_26_26/checkpoint/state_dict_12000.ckpt
```

## Commands Run

```bash
# Step 1 — pretrain on ukr_rus NM
bash scripts/cross_dataset/experiment_14/step1_submit_train_ukr_rus.sh
# checkpoint: /home1/singhama/gfm/prodigy/state/exp14_train1_ukr_rus_nm_23_04_2026_11_01_22/state_dict

# Step 2 — fine-tune on midterm NM
bash scripts/cross_dataset/experiment_14/step2_submit_finetune_midterm.sh \
  /home1/singhama/gfm/prodigy/state/exp14_train1_ukr_rus_nm_23_04_2026_11_01_22/state_dict

# Step 3 — eval on covid (NM + LP + PL, shots=1,5,10)
bash scripts/cross_dataset/experiment_14/step3_submit_eval_covid.sh \
  /home1/singhama/gfm/prodigy/state/exp14_train2_ukr_rus_nm_to_midterm_nm_23_04_2026_13_26_26/checkpoint/state_dict_12000.ckpt
```
