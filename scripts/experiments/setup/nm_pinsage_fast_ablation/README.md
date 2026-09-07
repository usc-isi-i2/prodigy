# Fast PinSAGE versus GraphSAGE ablation

This experiment changes only neighborhood selection and aggregation while retaining the
latest fast PRODIGY NM protocol: the shared all-eight graph, 40k optimizer steps, two-hop
context budget, one-hop NM positives, balanced source sampling, model dimensions, and seed.

The four paired source sets are COVID (largest graph), election2020-political (highest
average degree), TwiBot-20 (bot-domain relevance), and the all-eight mixture with TwiBot-20
left out. Every source set trains one historical GraphSAGE control and one PinSAGE arm.

PinSAGE uses 64 two-step walks from each episode center. Repeated visits are reduced to
counts; up to 100 distinct nodes become tokens, and normalized counts weight the messages
into the center. The NM-positive random walk remains a separate one-hop operation.

Run a dry plan and a bounded smoke before the 40k launch:

```bash
DRY_RUN=1 bash scripts/experiments/setup/nm_pinsage_fast_ablation/run_train_tucker.sh
SMOKE_STEPS=200 bash scripts/experiments/setup/nm_pinsage_fast_ablation/run_train_tucker.sh
```

`GPUS` is a space-separated list and `MODELS_PER_GPU` controls concurrency. For
example, `GPUS="0 1" MODELS_PER_GPU=4` runs all eight jobs while GPUs 2-3 are occupied.

Use a fresh run directory for the full launch. On Tucker it belongs in a detached tmux
session after checking `tmux ls`, GPU processes, RAM, and `/dev/shm`. Only GPUs 0-3 are
allowed.

After all eight terminal checkpoints exist, run the paired 10-shot classification
evaluation on the four labeled targets with matching inference samplers:

```bash
TRAIN_RUN_DIR=/dataMeR1/phil/gfm/prodigy-pinsage/log/pinsage_fast_<timestamp> \
  bash scripts/experiments/setup/nm_pinsage_fast_ablation/run_classification_eval_tucker.sh
```
