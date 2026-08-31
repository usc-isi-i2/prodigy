# RQ1 native-classification pilot

This pilot compares leave-one-family-out neighbor-matching initialization with
random initialization while retaining PRODIGY's native `S,U,M` classifier.

- Targets: COVID-19, Election 2020, Ukraine–Russia, and TwiBot-20.
- Budgets: 10, 100, and 1,000 labeled training nodes per class.
- Arms: pretrained and scratch, paired on the deterministic seed-0 label pool and
  episode stream.
- Training: 2-way, 5-shot, 5-query native classification episodes.
- Selection: raw-best validation ROC-AUC with patience 3.
- Test: evaluated once from the validation-selected checkpoint.
- Tracking: W&B online when launched with `WANDB_MODE=online`; local checkpoints,
  logs, metrics, hashes, and result JSONs remain canonical.

Run on Tucker GPUs 2 and 3:

```bash
WANDB_MODE=online GPUS_TEXT="2 3" SLOTS_PER_GPU=2 \
  bash scripts/experiments/setup/rq1_native_cls_pilot/run_all_tucker.sh
```
