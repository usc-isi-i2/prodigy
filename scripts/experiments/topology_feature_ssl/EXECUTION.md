# Execution tracker — topology_feature_ssl

Operational companion to `README.md` (the plan). Tracks what is **built** vs
**pending**, and gives the exact Tucker commands. Heavy jobs (pretrain/eval) are
launched by the user; this repo only ships the scripts.

## Build status

| Piece | Kind | Status |
|---|---|---|
| Free preview (nm vs fp on regression) | config-only, existing ckpts | ✅ built |
| B0 — control NM | config-only | ✅ built (`configs/B0.yaml`) |
| B1 — feature-shortcut corruption (`NR0.3`) | config-only | ✅ built (`configs/B1.yaml`) |
| Per-arm eval sweep (reg + slp + pl) | config-only | ✅ built (`run_eval_sweep.sh`) |
| 3-way merged name registered in loaders | code | ✅ done (trainer.py, data_loader_wrapper.py) |
| Diagnostics: 2×2 ablation + capability probes + leakage baseline | code | ⏳ pending (task #4) |
| E1 — directed structural input features | code | ⏳ pending (task #5) |
| E2 — expressive directed aggregator | code | ⏳ pending (task #6) |
| E3 — masked feature reconstruction | code (fp exists; refine) | ⏳ pending (task #7) |
| E4 — multi-task MFR ⊕ dir-LP ⊕ structural | code | ⏳ pending (task #8) |
| T1/T2/T3 tables + notebook | code | ⏳ pending (task #9) |

## Prerequisites (once, on Tucker)

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate prodigy
cd /dataMeR1/phil/gfm/prodigy

# 1. Build the 3-way merged pretrain corpus (assigns per-node graph_id, required
#    for within-source sampling). Skip if the .pt already exists.
ls -la /dataMeR1/phil/data/merged/graphs/ukr_rus_covid_midterm_retweet_graph.pt || \
  python3 scripts/graph_construction/merge_disjoint_graph_pt.py \
    scripts/graph_construction/merge_ukr_rus_covid_midterm_3way.yaml

# 2. Enrich eval graphs with benchmark targets (node_targets + static views) so
#    regression / static-LP have something to score. Skip if already enriched.
DATA_ROOT=/dataMeR1/phil/data bash scripts/graph_construction/enrich_all_graphs.sh
```

## Step 0 — Free preview (run first; no new pretrain)

Reads the existing covid nm/fp checkpoints; pre-validates E3 for ~zero cost.

```bash
bash scripts/experiments/topology_feature_ssl/run_free_preview.sh --gpus 0,1
```

Prints a per-(dataset, target) NM-vs-FP Spearman table + a `mean(fp-nm)` verdict.
mean(fp-nm) > 0 ⇒ a generative objective already helps regression ⇒ E3 pre-validated.

## Step 1 — B0 / B1 pretrains (the two config-only arms)

Smoke-test the flag path first (tiny, ~1 min), confirm the within-source strata
line prints, then launch the full 120k-episode pretrains.

```bash
cd /dataMeR1/phil/gfm/prodigy
ARM_DIR=scripts/experiments/topology_feature_ssl

# smoke (look for: "Neighbor sampling graph_id strata (confine-to-one-source): ...")
DRY_RUN=0 bash $ARM_DIR/train_arm_tucker.sh B0 --device 0 --epochs 1 \
  -ds_cap 20 -eval_step 20 -ckpt_step 20 --prefix tfssl_B0_smoke

# full pretrains (one GPU each; run in tmux)
tmux new-session -d -s tfssl_B0 'bash -lc "bash scripts/experiments/topology_feature_ssl/train_arm_tucker.sh B0 --device 0"'
tmux new-session -d -s tfssl_B1 'bash -lc "bash scripts/experiments/topology_feature_ssl/train_arm_tucker.sh B1 --device 1"'
```

## Step 2 — Eval sweep for B0 / B1

```bash
cd /dataMeR1/phil/gfm/prodigy
# build the arm-keyed model list from the trained checkpoints
STATE_DIR=/dataMeR1/phil/gfm/prodigy/state ARMS="B0 B1" \
  bash scripts/experiments/topology_feature_ssl/make_model_list.sh

# frozen-encoder benchmark: reg (10-shot) + static-LP (0-shot) + classification
tmux new-session -d -s tfssl_eval 'bash -lc "MODEL_LIST=scripts/experiments/topology_feature_ssl/model_list.txt bash scripts/experiments/topology_feature_ssl/run_eval_sweep.sh --gpus 0,1,2,3 > /tmp/tfssl_eval.log 2>&1"'
```

Results land (keyed by `model` = arm) in
`scripts/plotting/{node_regression,static_link_prediction}/data/*.csv`.

Diagnostics (2×2 ablation, capability probes) and arms E1–E4 are built in the
later tasks; their commands will be appended here as they land.
