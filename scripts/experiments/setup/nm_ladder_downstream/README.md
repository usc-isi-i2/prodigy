# NM ladder — downstream tasks

**Status:** run 2026-07-27; **regression column re-scored 2026-07-29** after the episodic
regression eval was found void. No training: all 21 encoders exist.

> **The regression numbers changed.** The first pass scored regression through the runner's
> episodic `task_name=regression` path. That path predicts through a `regression_head` that
> is in no checkpoint, loads with `strict=False` so it stays at random init, and
> `--eval_only` never takes an optimizer step — the number it reports is a fixed random
> projection of the frozen embedding (`setup/regression_probe_repair/README.md`, found
> 2026-07-28, eleven hours after the first figures were built). Regression now comes from
> `run_reg_probe_sweep.sh`, a frozen-encoder ridge probe fitted on the support set. Static
> LP and classification are unaffected: static LP never used the runner, and classification
> has no equivalent defect.

## What we are doing

The interpolation ladder has only ever been scored on **neighbor matching**, the same
objective it was pretrained with. Its headline —

> a test graph's AUC stays flat at its zero-shot transfer level until its graph enters the
> training merge, then jumps and holds

— is therefore a statement about NM AUC, not about anything a downstream user would run.
This experiment re-scores every rung on the tasks the graphs actually support, so the
staircase can be claimed (or not) as a property of the *representation* rather than of the
pretraining metric.

Confirmed before writing any of this: no `nm_ladder_*`, `*_wb`, or `nm_ss_*` row exists in
`analysis/{node_classification,node_regression,static_link_prediction}/data/`. The ladder
is genuinely NM-only today.

## The question, stated so it can fail

For each downstream task and each test graph, is the entry-aligned Δ (metric just after
that graph's source enters the merge − metric just before) positive, the way it was for NM
in `nm_ladder_order_robustness` (21/21 events, p = 4.8e-7)?

Three outcomes, all publishable:

- **Δ > 0 across tasks.** Adding a domain to pretraining buys measurable downstream
  performance on that domain. The ladder becomes a mixture-design result, not an NM result.
- **Δ ≈ 0 downstream while Δ > 0 on NM.** The staircase is an artifact of scoring with the
  pretraining objective — NM AUC measures how well the encoder solves NM on in-distribution
  neighborhoods, which is not the same as a useful representation. This would qualify a lot
  of the program's existing NM-based conclusions.
- **Δ mixed by task.** The feature-channel tasks (regression, classification) and the
  topology task (static LP) separate — which is the two-channels hypothesis the paper
  program is built on, measured on a cleaner design than before.

## Design

**24 rows, 21 encoders, 0 new trainings.** Rows are the full 3-order grid from
`nm_ladder_order_robustness-jul_23` (order A = published topical, B = donor-strength
descending, C = its reverse). Rung 8 is order-invariant, order B's rung 2 has the same
source *set* as order A's rung 2, and rung 1 of any order is that graph's single-source
specialist, so 24 rows are served by 21 distinct checkpoints. `make_model_list.py` reuses
`make_configs.plan()` to derive that mapping rather than restating it.

All 21 are at the **matched-40k** checkpoint in `/dataMeR1/phil/gfm/prodigy/state/`, the
same budget the NM table was read at — verified present 2026-07-27.

**Tasks and their eligible graphs** (from `docs/graph_catalog.json`):

| task | graphs | protocol | jobs |
|---|---|---|---|
| node regression | ukraine, covid, midterm, twibot20 | 10-shot, log1p, 3 targets, **frozen-encoder ridge probe** | 4 graph passes |
| node classification | covid-political, election2020, ukraine-susp, twibot20 | 10-shot | 84 |
| static link prediction | ukraine, covid, midterm, twibot20, hongkong | pair-conditioned, degree-matched negatives | 5 graph passes |

Shots, targets and transform are copied from `multitask_ssl_corpora/run_eval_sweep.sh`
verbatim, so the ladder rows land in the shared CSVs directly comparable to the existing
`cov_*` / `all8_*` / `B0` / `B1` / `E1` arms.

**Neither regression nor static LP goes through the runner any more.** Both episodic paths
are void, and both replacements invert the loop the same way: load the graph once, build
**one shared evaluation set**, score all 21 encoders against it. Every rung therefore sees
identical support/query nodes (regression) and identical positive/negative pairs (static
LP), and the floors are computed on those same items. 4 + 5 graph passes replace 252 + 105
runner jobs. Only the 84 classification jobs still use the runner.

For regression that floor is `__features_only__`: a ridge probe on the raw 768-d input
features, same episodes. It is the line an encoder must clear to be carrying anything the
inputs did not already carry — and on the old void path, encoders sat *below* it on exactly
the targets features predict best, which was the first sign something was wrong.

**Static LP does not go through the runner.** Its episodic `slp` path is void — center-blind
scoring, frozen random prototypes, degree-confounded negatives (AGENTS.md;
`analysis/multitask_ssl/FINDINGS_rescore.md`). `run_pair_lp_sweep.sh` drives
`scripts/eval/pair_link_sweep.py` instead, which loads each graph once and scores all 21
encoders against **one shared pair set**, with CN / Adamic-Adar / preferential-attachment /
Jaccard / raw-feature-cosine floors computed on that same set.

**Temporal LP is out of scope.** It has the same three defects and was never rescored;
including it would add rows nobody could cite. Repairing it is separate work.

## Known confound, stated up front

Rung-1 rows are single-source **specialists**, which beat any merged model on their own
graph (the dilution effect: NM cost +.006–.039 in-domain, largest on small/topical graphs).
Exclude rung-1 entries when quoting post-entry stability, exactly as the NM analysis did.

## Run order (Tucker)

Give this its own worktree — the checkpoints are referenced by absolute path, so the sweep
does not need to run from the checkout that trained them, and an isolated tree cannot have
its code pulled out from under a running job.

```bash
cd /dataMeR1/phil/gfm && git -C prodigy fetch origin experiment/nm-ladder-downstream
git -C prodigy worktree add ../prodigy-nmld experiment/nm-ladder-downstream
```

```bash
cd /dataMeR1/phil/gfm/prodigy-nmld && tmux new-session -d -s nmld_pipeline 'export PATH="/home/mhchu/miniconda3/bin:$PATH"; bash scripts/experiments/setup/nm_ladder_downstream/run_pipeline_tucker.sh'
```

Watch one file:

```bash
cat /dataMeR1/phil/gfm/prodigy-nmld/scripts/experiments/setup/nm_ladder_downstream/run_logs/pipeline_status.txt
```

Phases: `resolve → smoke → benchmark → pair_lp → assemble`. **Smoke is a hard stop** — it
scores one encoder on the smallest graph and asserts the evaluator's validity reads
(`leakage_edges=0`, `endpoint_sensitivity≈1`, `endpoint_permutation_auc≈0.5`). A
checkpoint/architecture mismatch would otherwise yield 336 confidently-wrong rows. Rerun a
single phase with `ONLY=pair_lp`.

Individual pieces, if you would rather drive them by hand:

```bash
STATE_DIR=/dataMeR1/phil/gfm/prodigy/state python3 scripts/experiments/setup/nm_ladder_downstream/make_model_list.py
```

```bash
MODEL_LIST=scripts/experiments/setup/nm_ladder_downstream/model_list.txt bash scripts/experiments/setup/nm_ladder_downstream/run_eval_sweep.sh --gpus 0,1,2
```

```bash
GPU=0 bash scripts/experiments/setup/nm_ladder_downstream/run_pair_lp_sweep.sh
```

Regression, after the gate passes (the gate is CPU-only and takes seconds — do not skip it,
it is what proves the probe reproduces the published raw-feature floor):

```bash
bash scripts/experiments/setup/regression_probe_repair/run_gate.sh
```

```bash
GPU=0 bash scripts/experiments/setup/nm_ladder_downstream/run_reg_probe_sweep.sh
```

Split across two GPUs by dataset if you want the wall-clock back — the small graphs finish
while covid19 is still loading, and the four dataset passes write to separate files:

```bash
DATASETS=midterm,twibot20 GPU=0 bash scripts/experiments/setup/nm_ladder_downstream/run_reg_probe_sweep.sh
```

```bash
DATASETS=ukr_rus_twitter,covid19_twitter GPU=1 bash scripts/experiments/setup/nm_ladder_downstream/run_reg_probe_sweep.sh
```

```bash
python3 scripts/experiments/setup/nm_ladder_downstream/assemble_downstream_tables.py
```

`DRY_RUN=1` previews either sweep without touching a GPU; `make_model_list.py --dry-run`
prints the 24-row plan and resolves nothing.

**GPU etiquette:** checked 2026-07-27 — `rdorn` runs a vLLM tensor-parallel worker pinned
across GPUs **2 and 3** (~76 GB each), so only 0 and 1 are actually free. The pipeline
defaults to `EVAL_GPUS=0,1`. This moves; check before launching and override:

```bash
ssh tucker nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv
```

## Deliverables

Into `analysis/nm_ladder_downstream/data/`:

- `nm_ladder_downstream_long.csv` — one row per (order, rung, task, dataset, target,
  metric), carrying `in_merge` and `rel_to_entry` so the three orders overlay on a common
  entry-aligned axis. Same shape as the NM `_long` table, so the event-study figure code
  transfers.
- `nm_ladder_downstream_{reg,pl,slp}.csv` — wide, 24 rows × test-graph columns, primary
  metric only (spearman / roc_auc / auc).
- `nm_ladder_downstream_slp_floors.csv` — the heuristic floors each static-LP number must
  be read against.
- `nm_ladder_downstream_reg_floors.csv` — the raw-feature floor each regression number must
  be read against, per (dataset, target).
- `data/pair_lp/<dataset>__pair_lp.csv` — the raw pair-evaluator output, including the
  validity columns.
- `data/reg_probe/<dataset>__reg_probe.csv` — the raw probe output, including the
  `__features_only__` floor rows and `n_labeled` / `n_pred`.

Analysis and findings go in `analysis/nm_ladder_downstream/`, not here.

## Cost

336 runner jobs plus 5 pair-evaluator graph passes, no training. The NM sweep did 32 jobs
in ~17 min on 4 GPUs; scaled up and on 3 GPUs this is an overnight run rather than a
coffee break. The two big pair-LP passes (ukraine, covid) dominate the tail and run serial
on purpose — the adjacency build is this repo's memory peak.
