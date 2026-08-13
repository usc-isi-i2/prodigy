# Fixed-exposure NM graph ladder at 2 hops

This experiment complements the matched-40k 2-hop ladder by holding expected
per-source exposure fixed at 10,000 neighbor-matching episodes. Total compute grows
with the number of active sources:

| Rung | Sources | Training steps |
|---:|---:|---:|
| 1 | 1 | 10,000 |
| 2 | 2 | 20,000 |
| 3 | 3 | 30,000 |
| 4 | 4 | 40,000 |
| 5 | 5 | 50,000 |
| 6 | 6 | 60,000 |
| 7 | 7 | 70,000 |
| 8 | 8 | 80,000 |

The setup branches from `codex/pretrain-saturation-nhop2` at `72c5e277`, the
completed fair-2-hop saturation revision. It does not modify the matched-40k ladder or
reuse any of its model prefixes, state directories, or model lists.

## Controlled protocol

Every training and evaluation process uses the fair 2-hop sampler registered by the
saturation experiment:

- two extracted hops with fanouts `9,9`;
- a hard limit of 101 nodes per subgraph;
- one-hop NM positive walks;
- the unchanged `256 · S,U,M` GraphSAGE encoder; and
- seed 0, 30-way/3-shot/4-query NM episodes.

Each rung samples its active source uniformly with
`neighbor_sampling_episode_source_weighting: balanced`. With `r` active sources and
`r × 10,000` total episodes, every source therefore has an expected exposure of 10,000
episodes. This fixes exposure in expectation; it does not impose a deterministic quota
on each source's realized random count.

All rungs read the existing disjoint all8 artifact and restrict eligible `graph_id`s.
Its source components have no cross-source edges, so a sampled neighborhood cannot
leave the selected source component.

## Scope and run count

The manifest contains 24 rows: three orders by eight rungs. Duplicate source sets reuse
the same new fixed-exposure model, leaving 21 unique training runs:

| Phase | Unique models | Purpose |
|---|---:|---|
| `A` | 8 | published topical order |
| `robustness` | 13 | remaining unique sets for orders B/C |
| `all` | 21 | both phases |

Order B rung 2 reuses order A rung 2, and all order-8 rows reuse one all8 model. Reuse
is valid because duplicate source sets have the same cardinality and thus the same
target step.

## Files and checks

- `make_configs.py` owns the orders, reuse plan, rung budgets, generated configs, and
  `manifest.tsv`.
- `run_all_train_tucker.sh` trains `smoke`, `A`, `robustness`, or `all`.
- `make_model_list.py` resolves every prefix at its own rung-specific final checkpoint.
- `eval_ladder_tucker.sh` locks the full `2 / 9,9 / 101 / walk=1` sampler tuple.
- Results and plots should go under a new matching analysis folder only after evidence
  exists; this setup does not create an empty analysis shell.

Verify generated files and inspect the launch plan locally:

```bash
python3 scripts/experiments/setup/nm_ladder_fixed_exposure_nhop2/make_configs.py --check
DRY_RUN=1 PHASE=A GPUS="0" \
  bash scripts/experiments/setup/nm_ladder_fixed_exposure_nhop2/run_all_train_tucker.sh
```

## Tucker isolation

Use a dedicated Tucker worktree. Check `tmux ls` and `git worktree list` first; never
pull or switch a worktree that owns a running job. Name the branch explicitly because
Tucker upstreams are not reliable:

```bash
cd /dataMeR1/phil/gfm
git -C prodigy fetch origin codex/nm-ladder-fixed-exposure-nhop2
git -C prodigy worktree add -b codex/nm-ladder-fixed-exposure-nhop2 \
  ../prodigy-nmlfxh2 origin/codex/nm-ladder-fixed-exposure-nhop2
cd /dataMeR1/phil/gfm/prodigy-nmlfxh2
git config core.hooksPath .githooks
```

State and logs then remain under that worktree's own `state/` and `log/`. Do not train
or evaluate from another checkout.

## Resource smoke

The default launcher uses one owned GPU because each process loads the approximately
111 GB all8 artifact. Before increasing parallelism, inspect GPUs 0-3 and host RAM.
GPUs 4-7 are not ours.

```bash
cd /dataMeR1/phil/gfm/prodigy-nmlfxh2
nvidia-smi
free -h

tmux new-session -d -s nmlfxh2_smoke \
  'export PATH="/home/mhchu/miniconda3/bin:$PATH"; \
   PHASE=smoke GPUS="0" bash scripts/experiments/setup/nm_ladder_fixed_exposure_nhop2/run_all_train_tucker.sh \
   > scripts/experiments/setup/nm_ladder_fixed_exposure_nhop2/run_logs/smoke_orchestrator.log 2>&1'
```

Require finite loss, an honest `state_dict_20.ckpt`, and acceptable memory/step time
before launching the ladder.

## Training

Dry-run and then launch canonical order A:

```bash
DRY_RUN=1 PHASE=A GPUS="0" \
  bash scripts/experiments/setup/nm_ladder_fixed_exposure_nhop2/run_all_train_tucker.sh

tmux new-session -d -s nmlfxh2_A \
  'export PATH="/home/mhchu/miniconda3/bin:$PATH"; \
   PHASE=A GPUS="0" bash scripts/experiments/setup/nm_ladder_fixed_exposure_nhop2/run_all_train_tucker.sh \
   > scripts/experiments/setup/nm_ladder_fixed_exposure_nhop2/run_logs/order_A_orchestrator.log 2>&1'
```

After order A, optionally train only the 13 remaining unique B/C models:

```bash
DRY_RUN=1 PHASE=robustness GPUS="0" \
  bash scripts/experiments/setup/nm_ladder_fixed_exposure_nhop2/run_all_train_tucker.sh

tmux new-session -d -s nmlfxh2_BC \
  'export PATH="/home/mhchu/miniconda3/bin:$PATH"; \
   PHASE=robustness GPUS="0" bash scripts/experiments/setup/nm_ladder_fixed_exposure_nhop2/run_all_train_tucker.sh \
   > scripts/experiments/setup/nm_ladder_fixed_exposure_nhop2/run_logs/orders_BC_orchestrator.log 2>&1'
```

The terminal-save fix makes `epochs: rung × dataset_len_cap: 10000` land exactly on
the intended 10k-80k final checkpoints. Do not add an extra epoch workaround.

## Evaluation

Resolve each model at its own final step and dry-run evaluation from the same worktree:

```bash
python3 scripts/experiments/setup/nm_ladder_fixed_exposure_nhop2/make_model_list.py \
  --phase A --state-dir /dataMeR1/phil/gfm/prodigy-nmlfxh2/state
PHASE=A GPUS="0" \
  bash scripts/experiments/setup/nm_ladder_fixed_exposure_nhop2/eval_ladder_tucker.sh --dry-run
```

For a complete three-order audit, `make_model_list.py --phase all` must resolve all 21
rung-specific checkpoints. The executed 2026-08-02 sweep intentionally covered Orders A
and C only: 15 physical models (A1–A8 and C1–C7; C8 reuses A8), eight evaluation graphs,
120/120 completed jobs, and zero failures. Order B remains deferred. Results and the
two-order analysis live in
[`nm_ladder_fixed_exposure_nhop2` analysis](../../analysis/transfer/ablations/source_exposure/nm_ladder_fixed_exposure_nhop2/).
