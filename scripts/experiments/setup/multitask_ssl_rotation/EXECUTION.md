# Execution tracker — multitask_ssl_rotation

Operational companion to `README.md` (the plan). Tracks what is **built** vs
**pending** and gives the exact Tucker commands. Heavy jobs (pretrain/eval) are
launched by the user; this repo ships the code + scripts.

## Build status

| Piece | Kind | Status |
|---|---|---|
| `nm_fp_cl` rotation task (per-episode sampler/aug/loss) | code | ✅ built |
| — params: `--mix_task_counts`, `--mix_cl_aug`, `mix` alias | code | ✅ `experiments/params.py` |
| — loader branch + per-task aug map | code | ✅ `data/covid19_twitter.py` |
| — Collator: per-episode aug + `mix_is_fp` tag | code | ✅ `data/dataloader.py` |
| — trainer: aux head + per-episode fp loss dispatch | code | ✅ `experiments/trainer.py` |
| NM / CL / FP single-task controls | config | ✅ `configs/{NM,CL,FP}.yaml` |
| MIX rotation arm | config | ✅ `configs/MIX.yaml` |
| Train / model-list / eval-sweep scripts | scripts | ✅ built |
| Smoke test passed on Tucker | run | ✅ done |
| Pretrains (NM/CL/FP/MIX), 1 seed, 40k | run | ✅ done (worktree, best-val=30k) |
| Eval sweep + T1 table | run | ✅ done → **[FINDINGS.md](../../analysis/objectives/multitask/multitask_ssl/FINDINGS.md)** |
| Aggregation (`aggregate_results.py`) | code | ✅ built |
| Multi-seed hardening (NM/MIX ×3) | run | ⛔ scoped out (single-seed, 2026-07-10) |
| WHY-LP ablation (eval-time, no retrain) | run | ⏳ next (Step 3 below) |

**Result:** MIX (rotation) is the only generalist — near-best cls, 2nd reg, and the
**only** arm with real static-LP (AUC 0.76 vs ≤0.47 chance for every control),
consistent across all 4 LP datasets. Full reading in [FINDINGS.md](../../analysis/objectives/multitask/multitask_ssl/FINDINGS.md).

**Ops note (worktree):** the 1-seed run executed from the isolated worktree
`/dataMeR1/phil/gfm/prodigy-mtr` (branch `mtr-run`). `run_eval_sweep.sh` now derives
`--log-root` from `REPO_ROOT` (was hardcoded to the main tree — it silently parsed
the wrong logs from a worktree). Conda is broken in detached tmux there; launch via
env-python directly (see the run memory).

## Prerequisites (once, on Tucker) — same as topology_feature_ssl

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate prodigy
cd /dataMeR1/phil/gfm/prodigy

# 3-way merged pretrain corpus (skip if the .pt already exists — it does if you ran tfssl)
ls -la /dataMeR1/phil/data/merged/graphs/ukr_rus_covid_midterm_retweet_graph.pt

# eval graphs enriched with benchmark targets (skip if already enriched for tfssl)
DATA_ROOT=/dataMeR1/phil/data bash scripts/graph_construction/enrich_all_graphs.sh
```

## Step 0 — Smoke test FIRST (critical: validates the new rotation code)

The rotation touches shared trainer/loader/collator code and was **not** runnable
offline (no torch locally), so smoke-test the flag path before any full run. Tiny,
~1–2 min. **Verify all four checks below.**

```bash
cd /dataMeR1/phil/gfm/prodigy
ARM_DIR=scripts/experiments/multitask_ssl_rotation

# MIX smoke — exercises the rotation (nm + cl + fp episodes) end to end
DRY_RUN=0 bash $ARM_DIR/train_arm_tucker.sh MIX --device 0 --epochs 1 \
  -ds_cap 30 -eval_step 30 -ckpt_step 30 --prefix mtr_MIX_smoke 2>&1 | tee /tmp/mtr_smoke.log

# also smoke each control quickly (they use existing single-task paths)
for A in NM CL FP; do
  bash $ARM_DIR/train_arm_tucker.sh $A --device 0 --epochs 1 \
    -ds_cap 20 -eval_step 20 -ckpt_step 20 --prefix mtr_${A}_smoke
done
```

**Smoke acceptance (MIX):**
1. Log prints `nm_fp_cl rotation (per-episode): nm:cl:fp counts = 1:1:1, cl_aug=NZ0.2, fp_aug=NZ0.3`.
2. Training completes 30 steps with **no shape/key errors** (the rotation cycles all
   three tasks, so a per-task bug surfaces within ~3 steps).
3. Both loss modes are exercised — `train_loss` is finite throughout, and fp episodes
   produce a (negative-score) reconstruction loss while nm/cl produce a metric acc.
4. A checkpoint is written under `state/mtr_MIX_smoke_*/checkpoint/` and val runs
   (pure-NM monitor) without error.

If (1)–(4) pass, the rotation is wired correctly; delete the `*_smoke` runs and
proceed. If step 2 fails on `mix_is_fp`, confirm `batch_size: 1` (the trainer enforces
it) and that `graph.mix_is_fp` survived `.to(device)` (see trainer `_episode_is_fp`).

## Step 1 — Full pretrains (4 arms, one GPU each, ~1.5 hr/arm at 40k)

```bash
cd /dataMeR1/phil/gfm/prodigy
tmux new-session -d -s mtr_NM  'bash -lc "bash scripts/experiments/setup/multitask_ssl_rotation/train_arm_tucker.sh NM  --device 0"'
tmux new-session -d -s mtr_CL  'bash -lc "bash scripts/experiments/setup/multitask_ssl_rotation/train_arm_tucker.sh CL  --device 1"'
tmux new-session -d -s mtr_FP  'bash -lc "bash scripts/experiments/setup/multitask_ssl_rotation/train_arm_tucker.sh FP  --device 2"'
tmux new-session -d -s mtr_MIX 'bash -lc "bash scripts/experiments/setup/multitask_ssl_rotation/train_arm_tucker.sh MIX --device 3"'
```

Checkpoints land at 10k/20k/30k/40k under `state/mtr_<ARM>_<timestamp>/checkpoint/`.

## Step 2 — Eval sweep (frozen encoders; all graphs × all tasks)

```bash
cd /dataMeR1/phil/gfm/prodigy
# arm-keyed model list from the trained checkpoints (highest-step ckpt per arm)
STATE_DIR=/dataMeR1/phil/gfm/prodigy/state ARMS="NM CL FP MIX" \
  bash scripts/experiments/setup/multitask_ssl_rotation/make_model_list.sh

# frozen-encoder benchmark: reg (10-shot) + static-LP (0-shot) + classification (10-shot)
tmux new-session -d -s mtr_eval 'bash -lc "MODEL_LIST=scripts/experiments/multitask_ssl_rotation/model_list.txt bash scripts/experiments/setup/multitask_ssl_rotation/run_eval_sweep.sh --gpus 0,1,2,3 > /tmp/mtr_eval.log 2>&1"'
```

Results land (keyed by `model` = arm ∈ {NM,CL,FP,MIX}) in
`scripts/experiments/analysis/evaluation/task_tables/{node_regression,static_link_prediction,node_classification}/data/*.csv`.
Then aggregate into the T1 table + headline reading:

```bash
python scripts/experiments/analysis/objectives/multitask/multitask_ssl/aggregate_results.py \
  --plotting-root scripts/experiments/analysis/evaluation/task_tables     # prints T1, MIX−max(NM,CL,FP), min-bar, per-dataset LP
```

Reading recorded in [FINDINGS.md](../../analysis/objectives/multitask/multitask_ssl/FINDINGS.md). **1 seed** — done.

## Step 3 — WHY does rotation yield LP? (eval-time ablation, NO retraining)

The frozen MIX encoder already exists, so the *mechanism* of the emergent LP win is
probed by re-running **static-LP only** under the tfssl 2×2 ablations (the eval
runner supports them directly). Run from the worktree that holds the checkpoints:

```bash
cd /dataMeR1/phil/gfm/prodigy-mtr
STATE_DIR=$PWD/state ARMS="NM CL FP MIX" \
  bash scripts/experiments/setup/multitask_ssl_rotation/make_model_list.sh
ML=scripts/experiments/multitask_ssl_rotation/model_list.txt
RUNNER=scripts/eval/eval_ckpts_all_graph_tasks_tucker.py
COMMON="--model-list $ML --python python3 --data-root /dataMeR1/phil/data \
  --datasets midterm,ukr_rus_twitter,covid19_twitter,twibot20 --continue-on-error \
  --tasks slp --shots 0 --slp-n-query 4 --gpus 0,1,2,3"

python3 $RUNNER $COMMON                          # clean (already have this)
python3 $RUNNER $COMMON --ablate-edges rewire    # destroy real adjacency, keep node bag/features
python3 $RUNNER $COMMON --ablate-features permute # destroy feature content, keep adjacency
```

**Prediction (the "rotation taught adjacency" hypothesis):** MIX static-LP AUC
collapses toward chance under `--ablate-edges rewire` and **survives**
`--ablate-features permute`; the controls stay at chance in all conditions.

**Parse caveat:** ablated runs get an `_ablE` / `_ablP` tag
(`eval_MIX_to_<ds>_slp_ablE_0shot_<ts>`), which the current `parse_benchmark_eval_logs.py`
SLP regex does **not** match — read `log/<run>/data/metrics_test.json`
(`roc_auc`, `accuracy`) directly, or extend the parser's `SLP_RE` to accept the tag.

## Notes / gotchas

- All four arms are bio-768 / mean-SAGE, so the eval sweep needs **no** `STRUCTURAL`
  or `GNN_TYPE` args (unlike the tfssl E-arms). One shared 768 sweep covers all four.
- MIX requires `batch_size: 1` (enforced in the trainer). Do not raise it — with >1,
  a single batch could mix tasks and the per-episode fp loss dispatch would be wrong.
- `mtr_*` prefixes keep these arms' benchmark rows namespaced apart from
  `topology_feature_ssl`'s `B0/E1/...` rows in the shared plotting CSVs.
