# NM graph ladder at 2 hops

This is an isolated, compute-matched `n_hop=2` rerun of the matched-40k
neighbor-matching graph ladder. It is based on the sampler implementation and
registered protocol from branch `codex/pretrain-saturation-nhop2` at `3705bd5`.
It does not modify or write into the 1-hop setup or analysis folders.

The controlled change is sampled neighborhood radius. Matching the saturation
experiment, every train and eval uses:

- two extracted hops with fanouts `9,9`;
- a hard limit of 101 nodes per subgraph, matching the 1-hop effective ceiling;
- one-hop NM positive walks, preserving the original positive definition; and
- the unchanged `256 · S,U,M` GraphSAGE encoder.

The remaining protocol is 30-way/3-shot NM, seed 0, within-source balanced
episodes, and a 40,000-episode budget. `S2,U,M` would be a different architecture
experiment and is deliberately out of scope. Model prefixes use `h2m`; old or
partial literal-2-hop `h2` artifacts are never resolved.

## Scope and run count

The all-order table has 24 rows: 3 orders × 8 rungs. There are 21 unique source
sets and therefore 21 new 2-hop checkpoints:

| Phase | Unique models | Evaluations | Purpose |
|---|---:|---:|---|
| `A` | 8 | 64 | canonical published order |
| `robustness` | 13 | 104 | remaining unique sets needed by orders B/C |
| `all` | 21 | 168 | both phases together |

Order B rung 2 reuses the new 2-hop order-A rung 2 model. All three order-8 rows
reuse the new 2-hop all8 model. No 1-hop checkpoint is reused, including the
COVID and election single-source specialists.

Every rung reads the existing disjoint all8 artifact and restricts eligible
`graph_id`s with `neighbor_sampling_source_subset`. Since components have no
cross-source edges, a 2-hop neighborhood cannot leave its source component, so
this is equivalent to building a separate nested merge for every rung.

## Files

- `make_configs.py` owns the orders, unique-set plan, and generated configs.
- `manifest.tsv` maps all 24 order/rung rows to the 21 unique models.
- `configs/` contains explicit compute-matched 2-hop configs and one resource smoke config.
- `run_all_train_tucker.sh` trains `smoke`, `A`, `robustness`, or `all`.
- `make_model_list.py` pins evaluation to `state_dict_40000.ckpt`.
- `eval_ladder_tucker.sh` passes the full `2 / 9,9 / 101 / walk=1` sampler tuple
  to every evaluation process.
- Results and plots belong in `analysis/transfer/ablations/prodigy_nm/context_depth/nm_ladder_nhop2/`, never here.

Generated-file integrity check:

```bash
python3 scripts/experiments/setup/nm_ladder_nhop2/make_configs.py --check
```

## Tucker isolation

Use a dedicated Tucker worktree. First check `tmux ls` and `git worktree list`;
never pull or switch a worktree that has a live job. Once the branch is available
on the remote, a typical setup is:

```bash
cd /dataMeR1/phil/gfm
git -C prodigy fetch origin
git -C prodigy worktree add -b codex/nm-ladder-nhop2 \
  ../prodigy-nmlh2 origin/codex/nm-ladder-nhop2
cd /dataMeR1/phil/gfm/prodigy-nmlh2
git config core.hooksPath .githooks
```

All state and logs then remain under that worktree's own `state/` and `log/`.
Do not evaluate from the main Tucker checkout: these directories are per-worktree.

## 1. Resource smoke

The compute-matched sampler prevents the naive fanout-100 explosion, but the
stress config still exercises the relatively high-degree election2020 source.
It runs 20 training episodes with one loader worker and a separate `h2m_smoke`
prefix that analysis ignores.

Before launching, check that an owned GPU (0–3) is actually free and inspect host
RAM. GPUs 4–7 are not ours.

```bash
cd /dataMeR1/phil/gfm/prodigy-nmlh2
nvidia-smi
free -h

tmux new-session -d -s nmlh2_smoke \
  'export PATH="/home/mhchu/miniconda3/bin:$PATH"; \
   PHASE=smoke GPUS="0" bash scripts/experiments/setup/nm_ladder_nhop2/run_all_train_tucker.sh \
   > scripts/experiments/setup/nm_ladder_nhop2/run_logs/smoke_orchestrator.log 2>&1'
```

After it finishes, inspect peak GPU/host memory and throughput in the log. Then
resolve and dry-run its one election evaluation:

```bash
python3 scripts/experiments/setup/nm_ladder_nhop2/make_model_list.py \
  --phase smoke --state-dir /dataMeR1/phil/gfm/prodigy-nmlh2/state
PHASE=smoke GPUS="0" bash scripts/experiments/setup/nm_ladder_nhop2/eval_ladder_tucker.sh --dry-run
```

Run the evaluation without `--dry-run` only after the command and resource use
look correct.

## 2. Canonical order A

The launcher defaults to one GPU because every process loads the ~104 GB all8
artifact. Increase parallelism only if the smoke and current Tucker host-memory
state justify loading that graph more than once.

```bash
DRY_RUN=1 PHASE=A GPUS="0" \
  bash scripts/experiments/setup/nm_ladder_nhop2/run_all_train_tucker.sh

tmux new-session -d -s nmlh2_A \
  'export PATH="/home/mhchu/miniconda3/bin:$PATH"; \
   PHASE=A GPUS="0" bash scripts/experiments/setup/nm_ladder_nhop2/run_all_train_tucker.sh \
   > scripts/experiments/setup/nm_ladder_nhop2/run_logs/order_A_orchestrator.log 2>&1'
```

The current trainer writes an honest terminal `state_dict_40000.ckpt` after
`epochs:4 × dataset_len_cap:10000`. Do not restore the old `epochs:5` workaround;
after the terminal-save fix that would also train and save an unwanted 50k model.

Resolve and evaluate the completed canonical models:

```bash
python3 scripts/experiments/setup/nm_ladder_nhop2/make_model_list.py \
  --phase A --state-dir /dataMeR1/phil/gfm/prodigy-nmlh2/state
PHASE=A GPUS="0" bash scripts/experiments/setup/nm_ladder_nhop2/eval_ladder_tucker.sh --dry-run
PHASE=A GPUS="0" bash scripts/experiments/setup/nm_ladder_nhop2/eval_ladder_tucker.sh
```

## 3. Orders B/C (optional second phase)

After order A is complete, train only the 13 remaining unique source sets:

```bash
DRY_RUN=1 PHASE=robustness GPUS="0" \
  bash scripts/experiments/setup/nm_ladder_nhop2/run_all_train_tucker.sh

tmux new-session -d -s nmlh2_BC \
  'export PATH="/home/mhchu/miniconda3/bin:$PATH"; \
   PHASE=robustness GPUS="0" bash scripts/experiments/setup/nm_ladder_nhop2/run_all_train_tucker.sh \
   > scripts/experiments/setup/nm_ladder_nhop2/run_logs/orders_BC_orchestrator.log 2>&1'
```

Evaluate the 13 new B/C models; the assembler will combine these logs with the
already-evaluated A models. `make_model_list.py` chooses the newest run directory
that actually contains the requested checkpoint, so a newer failed retry cannot
hide an older complete run.

```bash
python3 scripts/experiments/setup/nm_ladder_nhop2/make_model_list.py \
  --phase robustness --state-dir /dataMeR1/phil/gfm/prodigy-nmlh2/state
PHASE=robustness GPUS="0" bash scripts/experiments/setup/nm_ladder_nhop2/eval_ladder_tucker.sh --dry-run
PHASE=robustness GPUS="0" bash scripts/experiments/setup/nm_ladder_nhop2/eval_ladder_tucker.sh

# Optional completeness audit: all 21 checkpoints must resolve.
python3 scripts/experiments/setup/nm_ladder_nhop2/make_model_list.py \
  --phase all --state-dir /dataMeR1/phil/gfm/prodigy-nmlh2/state
```

`PHASE=all` remains available for a one-shot fresh evaluation, but rerunning it
after A and robustness would duplicate 64 completed A evaluations.

## Assemble and analyze

Run from the same Tucker worktree that owns the logs:

```bash
python3 scripts/experiments/analysis/transfer/ablations/prodigy_nm/context_depth/nm_ladder_nhop2/assemble_results.py \
  --log-root /dataMeR1/phil/gfm/prodigy-nmlh2/log
```

This writes the new 2-hop tables under
`scripts/experiments/analysis/transfer/ablations/prodigy_nm/context_depth/nm_ladder_nhop2/data/` and, when the committed
1-hop order table is present, a paired 1-hop-vs-2-hop comparison table.

## Archived Tucker artifacts

`model_list_A_archived.txt` preserves the exact Order-A checkpoint paths. The
retired worktree's ignored files are in
`/dataMeR1/phil/gfm/artifacts/worktree_cleanup_20260805/nmlh2-ignored.tar`. The
full pre-cleanup branch snapshot is tagged
`archive/preservation-nm-ladder-nhop2-2026-08-05`.
