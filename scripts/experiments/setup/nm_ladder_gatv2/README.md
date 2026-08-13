# NM interpolation ladder — GATv2 background encoder

## Question and scope

Does the canonical eight-rung NM interpolation staircase survive when the
sampled-subgraph encoder is changed from the default GraphSAGE implementation to
PyG `GATv2Conv`?

This is a one-factor backbone replication. Every rung is trained from scratch
with `gnn_type: gat`, which this repository maps to `GATv2Conv`. The layer string
stays `S,U,M`: `S` delegates to `gnn_type`, while `M` remains the published
metagraph message-passing module. Changing `M` to the experimental GATv2 `W`
module would alter a second factor and is explicitly out of scope.

Order robustness and downstream-task probes are also out of scope. Run those as
follow-ups only after the canonical NM ladder is complete.

## Registered protocol

- Eight fresh models, canonical source order:
  `ukr_rus_twitter`, `covid19_twitter`, `midterm`, `covid_political`,
  `election2020`, `ukr_rus_suspended`, `twibot20`, `cp_hk_twitter`.
- Existing rung artifacts are reused read-only; no graphs are built or modified.
- 256-dimensional, one background layer, `S,U,M`, no augmentation, seed 0.
- Merged rungs use within-source episodes and balanced source sampling.
- Exactly 40,000 optimizer steps: `epochs:4` × `dataset_len_cap:10000`.
- Evaluation is NM 30-way / 3-shot on all eight single-source graphs, using the
  same fixed split-derived episodes as the GraphSAGE ladder.
- The evaluator must receive `--gnn-type gat`; otherwise it constructs GraphSAGE
  and cannot load the GATv2 state dict. `eval_ladder_tucker.sh` pins this.
- `make_model_list.sh` accepts only `state_dict_40000.ckpt` and refuses partial
  eight-model lists. Never select the highest checkpoint implicitly.

Primary read: entry deltas for rungs 4–8. The replication supports the staircase
mechanism if every primary delta is positive, the four non-Twibot primary deltas
exceed the historical .02 ambiguity scale, pre-entry columns stay approximately
flat, and post-entry gains persist. These counts are descriptive for one training
seed, not a confidence interval.

## Isolation and synchronization

Development branch: `codex/nm-ladder-gatv2`.

Give the Tucker run its own checkout; do not pull or switch branches in any
worktree with a running job. Before creating it, inspect `tmux ls` and
`git worktree list` in the main Tucker checkout. After this branch is pushed:

```bash
cd /dataMeR1/phil/gfm
git -C prodigy fetch origin codex/nm-ladder-gatv2
git -C prodigy worktree add ../prodigy-gatladder \
  -b codex/nm-ladder-gatv2 origin/codex/nm-ladder-gatv2
cd /dataMeR1/phil/gfm/prodigy-gatladder
git rev-parse --show-toplevel
git rev-parse HEAD
```

If the branch or worktree already exists, inspect it rather than recreating or
deleting it. All `state/` and `log/` paths below belong to this worktree.

## 1. Read-only and dry-run checks

```bash
cd /dataMeR1/phil/gfm/prodigy-gatladder
python3 scripts/experiments/setup/nm_ladder_gatv2/validate_configs.py
bash scripts/experiments/setup/nm_ladder_gatv2/check_inputs_tucker.sh
DRY_RUN=1 GPUS="0 1 2 3" \
  bash scripts/experiments/setup/nm_ladder_gatv2/run_all_train_tucker.sh
nvidia-smi
```

`check_inputs_tucker.sh` only stats the existing graph files. Stop if any input
is missing; do not silently substitute the all8 source-subset shortcut.

## 2. Technical smoke test

Use one available owned GPU. This writes a distinct 20-step smoke run and cannot
be mistaken for a registered rung:

```bash
export PATH="/home/mhchu/miniconda3/bin:$PATH"
bash scripts/experiments/setup/nm_ladder_gatv2/train_nm_tucker.sh \
  train_1src.yaml --device 0 --dataset_len_cap 20 --epochs 1 \
  --checkpoint_step 20 --workers 2 --prefix nm_ladder_gatv2_smoke
```

Check its log for `GATv2Conv`, finite loss, parameter count, peak GPU memory, and
`state_dict_20.ckpt`. This gate tests wiring only; its AUC is not evidence.

## 3. Train the eight registered rungs

The user normally launches long jobs. The command below puts conda on `PATH`
inside tmux, partitions eight configs round-robin, and runs one job at a time per
GPU. Re-check GPU ownership and availability immediately before launch.

```bash
mkdir -p scripts/experiments/setup/nm_ladder_gatv2/run_logs
tmux new-session -d -s nml_gatv2 \
  'export PATH="/home/mhchu/miniconda3/bin:$PATH"; \
   GPUS="0 1 2 3" bash scripts/experiments/setup/nm_ladder_gatv2/run_all_train_tucker.sh \
   > scripts/experiments/setup/nm_ladder_gatv2/run_logs/orchestrator.log 2>&1'
```

Monitor without changing the checkout:

```bash
tmux capture-pane -pt nml_gatv2 -S -80
tail -n 80 scripts/experiments/setup/nm_ladder_gatv2/run_logs/orchestrator.log
```

GATv2 may have different throughput and memory from GraphSAGE, so use the smoke
measurement rather than inheriting the old ladder's ETA.

## 4. Resolve and evaluate matched-40k checkpoints

```bash
STATE_DIR=/dataMeR1/phil/gfm/prodigy-gatladder/state \
  bash scripts/experiments/setup/nm_ladder_gatv2/make_model_list.sh

GPUS="0,1,2,3" DATA_ROOT=/dataMeR1/phil/data \
  bash scripts/experiments/setup/nm_ladder_gatv2/eval_ladder_tucker.sh --dry-run

GPUS="0,1,2,3" DATA_ROOT=/dataMeR1/phil/data \
  bash scripts/experiments/setup/nm_ladder_gatv2/eval_ladder_tucker.sh
```

This is 8 × 8 = 64 NM evaluations. The generated log directories begin with
`eval_nm_ladder_gatv2_r<N>_<N>src_to_...`.

## 5. Assemble and compare

```bash
python3 scripts/experiments/analysis/transfer/ablations/prodigy_encoder/nm_ladder_gatv2/analyze_results.py \
  --log-root /dataMeR1/phil/gfm/prodigy-gatladder/log
```

The analyzer refuses incomplete matrices and writes dedicated evidence under
`scripts/experiments/analysis/transfer/ablations/prodigy_encoder/nm_ladder_gatv2/data/`. Inspect raw metric JSONs
before committing generated data or writing `RESULTS.md`.

If a primary transition is null or reversed, do not relabel the seed-0 outcome.
Repeat the disputed adjacent rung pair with seeds 1 and 2 under distinct prefixes
before deciding whether the difference is training noise or backbone-specific.
