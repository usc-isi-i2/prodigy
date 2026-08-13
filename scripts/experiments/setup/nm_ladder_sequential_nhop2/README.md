# Sequential two-hop NM graph ladder

This experiment reruns the canonical eight-rung NM graph ladder with **blocked
sequential source exposure** instead of balanced interleaving. It is based on
`codex/pretrain-saturation-nhop2` at `3705bd5` and is isolated from the concurrently
running interleaved two-hop control.

## Controlled comparison

Each rung is an independent seed-0 training run from random initialization. It receives
exactly 40,000 optimizer steps, the same total budget as its interleaved control. Expected
per-source exposure is also matched: the 40k episodes are divided as evenly as integer
counts permit, but presented in contiguous source blocks.

| rung | canonical source order | block lengths |
|---:|---|---|
| 1 | ukraine | 40,000 |
| 2 | ukraine → covid | 20,000, 20,000 |
| 3 | ukraine → covid → midterm | 13,334, 13,333, 13,333 |
| 4 | + covid-political | 10,000 each |
| 5 | + election2020-political | 8,000 each |
| 6 | + ukraine-suspended | 6,667 × 4, then 6,666 × 2 |
| 7 | + twibot20 | 5,715 × 2, then 5,714 × 5 |
| 8 | + hongkong | 5,000 each |

The at-most-seven remainder episodes go to the earliest blocks. No graph is revisited
after its block ends. A rung stays in one training process, so model weights **and AdamW
optimizer state** remain continuous across transitions. Separate checkpoint-warmstarted
runs would reset optimizer state and are not equivalent.

The only intended intervention relative to the control is source presentation:

- control: one source chosen uniformly for each episode;
- sequential: the same per-source counts in one block per source.

## Locked fair-two-hop protocol

Every run uses `n_hop=2`, fanouts `9,9`, node limit 101, one-hop NM-positive walks,
`256 · S,U,M` GraphSAGE, 30-way/3-shot NM, seed 0, and an honest 40k terminal checkpoint.
All rungs load the disjoint all8 graph and select their prefix through `graph_id`.

`make_configs.py` is the source of truth for source order, block allocation, transition
checkpoints, model names, and `manifest.tsv`. Generated configs are committed.

```bash
python3 scripts/experiments/setup/nm_ladder_sequential_nhop2/make_configs.py --check
DRY_RUN=1 PHASE=all GPUS="0" \
  bash scripts/experiments/setup/nm_ladder_sequential_nhop2/run_all_train_tucker.sh
```

## Worktree isolation

The local implementation branch is `codex/nm-ladder-sequential-nhop2`. On Tucker, add a
dedicated worktree only after the branch is pushed. Do not change or pull the worktree
running the interleaved control.

```bash
cd /dataMeR1/phil/gfm
git -C prodigy fetch origin
git -C prodigy worktree add -b codex/nm-ladder-sequential-nhop2 \
  ../prodigy-nmlh2seq origin/codex/nm-ladder-sequential-nhop2
cd /dataMeR1/phil/gfm/prodigy-nmlh2seq
git config core.hooksPath .githooks
bash scripts/experiments/setup/nm_ladder_sequential_nhop2/check_inputs_tucker.sh
```

State and logs live in this worktree's own `state/` and `log/`. Never evaluate from a
different checkout unless absolute paths point back here.

## Smoke and launch

The smoke loads the all8 graph because it must verify an actual transition from the
Ukraine `graph_id` to COVID after ten steps. It therefore tests the new schedule, not just
the already-validated two-hop neighborhood budget. Inspect `tmux ls`, `nvidia-smi`, host
RAM, and `git worktree list` first. Only GPUs 0–3 are ours.

```bash
tmux new-session -d -s nmlh2seq_smoke \
  'export PATH="/home/mhchu/miniconda3/bin:$PATH"; \
   PHASE=smoke GPUS="0" bash scripts/experiments/setup/nm_ladder_sequential_nhop2/run_all_train_tucker.sh \
   > scripts/experiments/setup/nm_ladder_sequential_nhop2/run_logs/smoke_orchestrator.log 2>&1'
```

The smoke passes only when the log prints the exact schedule `ukr_rus:10, covid:10`, loss
stays finite, and checkpoints 0, 10, and 20 exist. Then launch the eight rungs. The default
is deliberately one GPU because each process loads the roughly 104 GB all8 artifact.
Parallelize only if current host RAM supports multiple copies.

```bash
tmux new-session -d -s nmlh2seq \
  'export PATH="/home/mhchu/miniconda3/bin:$PATH"; \
   PHASE=all GPUS="0" bash scripts/experiments/setup/nm_ladder_sequential_nhop2/run_all_train_tucker.sh \
   > scripts/experiments/setup/nm_ladder_sequential_nhop2/run_logs/orchestrator.log 2>&1'
```

The launcher propagates per-run failures. `SKIP="train_r3.yaml train_r4.yaml"` can omit
completed configs during a retry.

## Terminal evaluation

Transition checkpoints are retained for a later forgetting trajectory, but the registered
primary comparison evaluates terminal 40k checkpoints only: 8 rungs × 8 graphs = 64 jobs.

```bash
python3 scripts/experiments/setup/nm_ladder_sequential_nhop2/make_model_list.py \
  --state-dir /dataMeR1/phil/gfm/prodigy-nmlh2seq/state

GPUS="0" bash scripts/experiments/setup/nm_ladder_sequential_nhop2/eval_ladder_tucker.sh --dry-run
GPUS="0" bash scripts/experiments/setup/nm_ladder_sequential_nhop2/eval_ladder_tucker.sh
```

The eval wrapper explicitly passes the full `2 / 9,9 / 101 / walk=1` tuple. Do not compare
these models to the historical one-hop ladder as the schedule effect would be confounded
with neighborhood radius.

## Assemble

After the separate interleaved control has produced
`analysis/transfer/ablations/prodigy_nm/context_depth/nm_ladder_nhop2/data/nm_ladder_nhop2_long.csv`:

```bash
python3 scripts/experiments/analysis/transfer/ablations/prodigy_nm/source_schedule/nm_ladder_sequential_nhop2/assemble_results.py \
  --log-root /dataMeR1/phil/gfm/prodigy-nmlh2seq/log \
  --control-long /path/to/nm_ladder_nhop2_long.csv
python3 scripts/experiments/analysis/transfer/ablations/prodigy_nm/source_schedule/nm_ladder_sequential_nhop2/plot_comparison.py
```

The primary estimand is paired `AUC_sequential − AUC_interleaved` for the same rung and
test graph, summarized separately for the newest graph, incumbent graphs, and held-out
graphs. With one seed, report paired descriptive effects and signs, not confidence
intervals or seed variance.
