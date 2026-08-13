# Leakage-free fair-two-hop NM ladder

This experiment repeats the canonical eight-rung neighbor-matching graph ladder with a
real train/test edge split. It branches from `codex/pretrain-saturation-nhop2` at
`72c5e27`, uses the registered compute-matched two-hop sampler, and does not reuse the
historical ladder's artifacts, prefixes, state, logs, or analysis outputs.

## What “train/test split” means

The historical NM loaders used `train`, `val`, and `test` only to seed different random
episodes over the same adjacency. This experiment instead uses the graph artifacts'
seeded 85/15 static split over undirected edge pairs:

- training positives and all subgraph message passing use `static_background`;
- validation/test support and query positives use `static_holdout`;
- a held-out pair is absent from the background in both directions; and
- candidate centers must have at least seven distinct neighbors in the relevant view,
  enough for 3-shot + 4-query episodes. Every source must retain at least 30 eligible
  centers or the preflight fails.

This is transductive in nodes/features but inductive in the NM relation being scored.
It preserves the original ladder question—what happens when each graph enters the
pretraining mixture—without training on the evaluated positive edges.

## Locked protocol

- Canonical order: Ukraine, COVID, Midterm, COVID-political,
  Election2020-political, Ukraine-suspended, TwiBot-20, Hong Kong.
- Eight independently initialized rungs, each trained for exactly 40,000 episodes.
- Within-source episodes and uniform source selection among active sources.
- Seed 0, 30-way/3-shot/4-query NM.
- `n_hop=2`, fanouts `9,9`, node limit 101, one-hop NM walks.
- `256 · S,U,M` GraphSAGE, unchanged from the fair-two-hop control.

## Isolation

Local branch/worktree:

```text
codex/nm-ladder-train-test-nhop2
/private/tmp/prodigy-nmlsplit-h2
```

After the branch is pushed, create a separate Tucker worktree. Never pull or switch any
worktree with a live job:

```bash
cd /dataMeR1/phil/gfm
git -C prodigy fetch origin codex/nm-ladder-train-test-nhop2
git -C prodigy worktree add -b codex/nm-ladder-train-test-nhop2 \
  ../prodigy-nmlsplit-h2 origin/codex/nm-ladder-train-test-nhop2
cd /dataMeR1/phil/gfm/prodigy-nmlsplit-h2
git config core.hooksPath .githooks
```

## Build and validate the separate merge

The new merge preserves only `static_background` and `static_holdout` while dropping
incompatible edge features. Inputs with stored views use them. For older small artifacts
without views, the builder derives the same seed-0 85/15 undirected-pair split in memory;
their source files remain unchanged. The existing all-eight artifact is never overwritten.

```bash
python3 scripts/experiments/setup/nm_ladder_train_test_nhop2/check_inputs_tucker.py --sources
DRY_RUN=1 bash scripts/experiments/setup/nm_ladder_train_test_nhop2/build_split_merge_tucker.sh
bash scripts/experiments/setup/nm_ladder_train_test_nhop2/build_split_merge_tucker.sh
python3 scripts/experiments/setup/nm_ladder_train_test_nhop2/check_inputs_tucker.py
```

Graph construction uses `bio-embeddings-v001`; training/evaluation use `prodigy`.

## Smoke, then parallel training

First check `tmux ls`, `nvidia-smi`, `free -h`, and `git worktree list`. GPUs 0–3 are
ours; 4–7 are not. Do not overlap this with another four-copy all8 job.

```bash
DRY_RUN=1 PHASE=smoke GPUS="0" \
  bash scripts/experiments/setup/nm_ladder_train_test_nhop2/run_all_train_tucker.sh

tmux new-session -d -s nmlsplit_h2_smoke \
  'export PATH="/home/mhchu/miniconda3/bin:$PATH"; \
   PHASE=smoke GPUS="0" bash scripts/experiments/setup/nm_ladder_train_test_nhop2/run_all_train_tucker.sh \
   > scripts/experiments/setup/nm_ladder_train_test_nhop2/run_logs/smoke_orchestrator.log 2>&1'
```

Require finite loss, `state_dict_20.ckpt`, and a successful held-out smoke evaluation.
Then run eight models in two four-GPU waves:

```bash
DRY_RUN=1 PHASE=all GPUS="0 1 2 3" \
  bash scripts/experiments/setup/nm_ladder_train_test_nhop2/run_all_train_tucker.sh

tmux new-session -d -s nmlsplit_h2 \
  'export PATH="/home/mhchu/miniconda3/bin:$PATH"; \
   PHASE=all GPUS="0 1 2 3" bash scripts/experiments/setup/nm_ladder_train_test_nhop2/run_all_train_tucker.sh \
   > scripts/experiments/setup/nm_ladder_train_test_nhop2/run_logs/orchestrator.log 2>&1'
```

Use `SKIP="configs/train_r1.yaml ..."` for completed rungs on a retry.

## Terminal evaluation

```bash
python3 scripts/experiments/setup/nm_ladder_train_test_nhop2/make_model_list.py \
  --state-dir /dataMeR1/phil/gfm/prodigy-nmlsplit-h2/state
GPUS="0,1,2,3" bash scripts/experiments/setup/nm_ladder_train_test_nhop2/eval_ladder_tucker.sh --dry-run
GPUS="0,1,2,3" bash scripts/experiments/setup/nm_ladder_train_test_nhop2/eval_ladder_tucker.sh
```

The registered result is the complete 8×8 terminal-checkpoint table. Do not treat a
partial table or the training-view monitor as test evidence.

```bash
python3 scripts/experiments/analysis/transfer/ablations/train_test_separation/nm_ladder_train_test_nhop2/assemble_results.py \
  --log-root /dataMeR1/phil/gfm/prodigy-nmlsplit-h2/log
```
