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
| Smoke test passed on Tucker | run | ⏳ pending (do this FIRST) |
| Pretrains (NM/CL/FP/MIX) | run | ⏳ pending |
| Eval sweep + T1 table | run | ⏳ pending |

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
tmux new-session -d -s mtr_NM  'bash -lc "bash scripts/experiments/multitask_ssl_rotation/train_arm_tucker.sh NM  --device 0"'
tmux new-session -d -s mtr_CL  'bash -lc "bash scripts/experiments/multitask_ssl_rotation/train_arm_tucker.sh CL  --device 1"'
tmux new-session -d -s mtr_FP  'bash -lc "bash scripts/experiments/multitask_ssl_rotation/train_arm_tucker.sh FP  --device 2"'
tmux new-session -d -s mtr_MIX 'bash -lc "bash scripts/experiments/multitask_ssl_rotation/train_arm_tucker.sh MIX --device 3"'
```

Checkpoints land at 10k/20k/30k/40k under `state/mtr_<ARM>_<timestamp>/checkpoint/`.

## Step 2 — Eval sweep (frozen encoders; all graphs × all tasks)

```bash
cd /dataMeR1/phil/gfm/prodigy
# arm-keyed model list from the trained checkpoints (highest-step ckpt per arm)
STATE_DIR=/dataMeR1/phil/gfm/prodigy/state ARMS="NM CL FP MIX" \
  bash scripts/experiments/multitask_ssl_rotation/make_model_list.sh

# frozen-encoder benchmark: reg (10-shot) + static-LP (0-shot) + classification (10-shot)
tmux new-session -d -s mtr_eval 'bash -lc "MODEL_LIST=scripts/experiments/multitask_ssl_rotation/model_list.txt bash scripts/experiments/multitask_ssl_rotation/run_eval_sweep.sh --gpus 0,1,2,3 > /tmp/mtr_eval.log 2>&1"'
```

Results land (keyed by `model` = arm ∈ {NM,CL,FP,MIX}) in
`scripts/plotting/{node_regression,static_link_prediction,node_classification}/data/*.csv`.
Build the T1 table (README) as a table subtraction MIX − max(NM,CL,FP), scored by
`min(feature, topological)`.

## Notes / gotchas

- All four arms are bio-768 / mean-SAGE, so the eval sweep needs **no** `STRUCTURAL`
  or `GNN_TYPE` args (unlike the tfssl E-arms). One shared 768 sweep covers all four.
- MIX requires `batch_size: 1` (enforced in the trainer). Do not raise it — with >1,
  a single batch could mix tasks and the per-episode fp loss dispatch would be wrong.
- `mtr_*` prefixes keep these arms' benchmark rows namespaced apart from
  `topology_feature_ssl`'s `B0/E1/...` rows in the shared plotting CSVs.
