# Pretrain saturation with compute-matched two-hop context

Fresh replication of `pretrain_saturation`, isolated from its spliced historical
trajectories. Three corpora are trained from random initialization to exactly 40,000
optimizer steps and checkpointed at `0, 100, 500, 1k, 2k, 10k, 40k`.

Status: complete (3/3 trajectories, 76/76 classification evaluations, and all four
regression probes). Evidence and interpretation are in
[`pretrain_saturation_nhop2/FINDINGS.md`](../../analysis/transfer/ablations/prodigy_nm/saturation/pretrain_saturation_nhop2/FINDINGS.md).

## Controlled intervention

The historical `n_hop=1` sampler uses fanout 100 and therefore extracts at most about
101 nodes (center + neighbours) and 100 sampled edges per subgraph. A naive `n_hop=2`
setting branches again and can hit the 2,000-node cap, confounding context radius with up
to 20x more nodes.

This experiment instead sets:

- `n_hop: 2` for two-hop context;
- per-hop fanouts `9,9`, giving at most about 100 sampled nodes and 99 sampled edges;
- hard node limit 101, matching the one-hop effective ceiling; and
- `neighbor_matching_walk_hops: 1`, preserving direct-neighbour NM positives.

Thus the positive definition and approximate node/edge budget stay fixed; only context
radius changes. The architecture remains one background layer, `S,U,M`. Two-hop nodes
enter the global subgraph pooling readout, but this is not an `S2` experiment.

## Registered protocol

- Arms: `all8` (within-source, balanced), `ukr`, `covid`.
- Seed 0, GraphSAGE, 256 dimensions, `S,U,M`, no augmentation.
- Compute-matched `n_hop=2`, fanouts `9,9`, and node limit 101 in both pretraining and
  downstream embedding extraction.
- One-hop NM positive walks, matching the historical objective.
- Classification: the same four labelled graphs and fixed 10-shot episodes as the
  original saturation experiment.
- Regression: only the repaired frozen-encoder ridge probe, on `followers_count` and
  `account_age_days`. The runner's episodic regression path is void and is not invoked.
- Dedicated model keys (`sat_h2m_*`) and dedicated analysis outputs; shared benchmark
  CSVs are not edited.

The abandoned literal-h2 pilot used `sat_h2_*` keys and is not evidence. Its partial
checkpoints may remain in the worktree state directory, but the resolver only accepts
complete `sat_h2m_*` trajectories.

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
tmux new-session -d -s sat_h2m \
   'export PATH="/home/mhchu/miniconda3/bin:$PATH"; \
   GPUS="0 1 2" bash scripts/experiments/setup/pretrain_saturation_nhop2/run_all_train_tucker.sh \
   > scripts/experiments/setup/pretrain_saturation_nhop2/run_logs/orchestrator_h2m.log 2>&1'
```

Resolve complete trajectories. The resolver refuses missing checkpoints and requires all
three step-0 model states to be tensor-identical before collapsing them into one eval row.
It intentionally ignores serialization-only differences in the PyTorch archive bytes.

```bash
python3 scripts/experiments/setup/pretrain_saturation_nhop2/make_model_list.py
```

Then evaluate and assemble:

```bash
GPUS="0,1,2" bash scripts/experiments/setup/pretrain_saturation_nhop2/run_classification_sweep.sh
GPU=0 bash scripts/experiments/setup/pretrain_saturation_nhop2/run_reg_probe_sweep.sh
python3 scripts/experiments/analysis/transfer/ablations/prodigy_nm/saturation/pretrain_saturation_nhop2/analyze_results.py \
  --log-root "$PWD/log"
```

Split regression graph passes across GPUs with distinct datasets if useful; each dataset
writes a separate file. Always recheck `nvidia-smi`, `tmux ls`, and `git worktree list`
immediately before a launch.
