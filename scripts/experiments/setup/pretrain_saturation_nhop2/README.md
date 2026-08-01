# Pretrain saturation with literal `n_hop=2`

Fresh replication of `pretrain_saturation`, isolated from its spliced historical
trajectories. Three corpora are trained from random initialization to exactly 40,000
optimizer steps and checkpointed at `0, 100, 500, 1k, 2k, 10k, 40k`.

## What this intervention means

`n_hop` is shared by two mechanisms in this repository. Setting it to 2:

1. makes NM positives the endpoints of two-step random walks instead of direct
   neighbours (`NeighborTask` calls `neighbor_sampler.random_walk`); and
2. extracts two-hop subgraphs around every support/query node.

The architecture remains one background layer, `S,U,M`. Two-hop nodes enter the global
subgraph pooling readout, but this is not an `S2` two-layer-message-passing experiment.
The registered claim is therefore: **does downstream budget saturation persist when NM
uses two-step positives and two-hop sampled context?** It is not a context-only ablation.

## Registered protocol

- Arms: `all8` (within-source, balanced), `ukr`, `covid`.
- Seed 0, GraphSAGE, 256 dimensions, `S,U,M`, no augmentation.
- `n_hop=2` in both pretraining and downstream embedding extraction.
- Classification: the same four labelled graphs and fixed 10-shot episodes as the
  original saturation experiment.
- Regression: only the repaired frozen-encoder ridge probe, on `followers_count` and
  `account_age_days`. The runner's episodic regression path is void and is not invoked.
- Dedicated model keys (`sat_h2_*`) and dedicated analysis outputs; shared benchmark
  CSVs are not edited.

The sampler uses fanout 100 and a 2,000-node cap. A 30-way `(3 support + 4 query)` NM
episode contains 210 sampled subgraphs, so the dense-source smoke test is a hard resource
gate. Registered configs use two loader workers and the regression encoder batches 32
nodes at a time to bound host/GPU memory.

## Tucker workflow

Use a dedicated worktree and verify its branch and revision. Do not pull or switch it
while a job is running.

```bash
python3 scripts/experiments/setup/pretrain_saturation_nhop2/validate_configs.py
bash scripts/experiments/setup/pretrain_saturation_nhop2/check_inputs_tucker.sh
DRY_RUN=1 GPUS="0 1 2" \
  bash scripts/experiments/setup/pretrain_saturation_nhop2/run_all_train_tucker.sh
```

Run the 20-step dense-source smoke on one available owned GPU. It uses the standalone
election graph so the resource gate does not first load the 111 GB all8 artifact.

```bash
GPU=0 bash scripts/experiments/setup/pretrain_saturation_nhop2/run_smoke_tucker.sh
```

Only after finite loss, `state_dict_20.ckpt`, and acceptable memory/step time:

```bash
tmux new-session -d -s sat_h2 \
  'export PATH="/home/mhchu/miniconda3/bin:$PATH"; \
   GPUS="0 1 2" bash scripts/experiments/setup/pretrain_saturation_nhop2/run_all_train_tucker.sh \
   > scripts/experiments/setup/pretrain_saturation_nhop2/run_logs/orchestrator.log 2>&1'
```

Resolve complete trajectories. The resolver refuses missing checkpoints and requires all
three step-0 checkpoints to be byte-identical before collapsing them into one eval row.

```bash
python3 scripts/experiments/setup/pretrain_saturation_nhop2/make_model_list.py
```

Then evaluate and assemble:

```bash
GPUS="0,1,2" bash scripts/experiments/setup/pretrain_saturation_nhop2/run_classification_sweep.sh
GPU=0 bash scripts/experiments/setup/pretrain_saturation_nhop2/run_reg_probe_sweep.sh
python3 scripts/experiments/analysis/pretrain_saturation_nhop2/analyze_results.py \
  --log-root "$PWD/log"
```

Split regression graph passes across GPUs with distinct datasets if useful; each dataset
writes a separate file. Always recheck `nvidia-smi`, `tmux ls`, and `git worktree list`
immediately before a launch.
