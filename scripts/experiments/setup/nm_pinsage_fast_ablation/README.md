# Fast PinSAGE versus GraphSAGE ablation

This experiment changes only neighborhood selection and aggregation while retaining the
latest PRODIGY final-core NM protocol with shared-graph loading: the immutable all-nine
70/15/15 edge-split graph, 2,500 optimizer updates at batch size 4, learning rate 0.002,
two-hop context, one-hop NM positives, balanced source sampling, and checkpoints at
100/300/900/2,500, repeated with seeds 0, 1, and 2.

The four paired source sets are COVID (largest graph), election2020-political (highest
average degree), TwiBot-20 (bot-domain relevance), and the all-nine mixture with TwiBot-20
left out. Every source set trains paired GraphSAGE and PinSAGE arms for all three seeds
(24 physical models).

PinSAGE uses 64 two-step walks from each episode center. Repeated visits are reduced to
counts; up to 100 distinct nodes become tokens, and normalized counts weight the messages
into the center. The NM-positive random walk remains a separate one-hop operation.

Run a dry plan and a bounded smoke before the full 2,500-update launch:

```bash
DRY_RUN=1 bash scripts/experiments/setup/nm_pinsage_fast_ablation/run_train_tucker.sh
SMOKE_STEPS=200 bash scripts/experiments/setup/nm_pinsage_fast_ablation/run_train_tucker.sh
```

`GPUS` is a space-separated list and `MODELS_PER_GPU` controls concurrency. For
example, `GPUS="0 1" MODELS_PER_GPU=4` runs eight jobs concurrently and queues the
remaining jobs while GPUs 2-3 are occupied.

Use a fresh run directory for the full 2,500-update launch. On Tucker it belongs in a detached tmux
session after checking `tmux ls`, GPU processes, RAM, and `/dev/shm`. Only GPUs 0-3 are
allowed.

After all 24 terminal checkpoints exist, run the paired 10-shot classification
evaluation on the four labeled targets with matching inference samplers:

```bash
TRAIN_RUN_DIR=/dataMeR1/phil/gfm/prodigy-pinsage/log/pinsage_fast_<timestamp> \
  bash scripts/experiments/setup/nm_pinsage_fast_ablation/run_classification_eval_tucker.sh
```

For the held-out NM comparison against final core, evaluate the four completed seed-0
PinSAGE checkpoints on the frozen 512-episode test streams. The evaluator verifies both
published episode fingerprints before accepting a result:

```bash
GPUS="0 1 2 3" bash scripts/experiments/setup/nm_pinsage_fast_ablation/run_fixed_test_tucker.sh
```
