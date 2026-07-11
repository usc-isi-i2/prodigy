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
| Diagnostics: 2×2 rewired-edge ablation | code | ✅ built (`run_2x2_ablation.sh`, `parse_2x2.py`) |
| Diagnostics: leakage baseline | code | ✅ built (`leakage_baseline.py`) |
| Diagnostics: capability probes | code | ✅ built (`make_probe_graphs.py`, `run_capability_probes.sh`) |
| E1 — directed structural input features | code | ✅ built (`configs/E1.yaml`, `--structural_features directed3`) |
| E2 — expressive directed aggregator | code | ✅ built (`configs/E2.yaml`); ⏳ **true-40k rerun + matched eval pending** (see Step 4) |
| E2b — drop-BN encoder retry | config + eval flag | ✅ built (`configs/E2b.yaml` `no_bn_encoder:true`; eval `--no-bn-encoder`); ⏳ run pending (Step 5) |
| E3 — masked feature reconstruction | code (fp exists; refine) | ⏳ pending (task #7) |
| E4 — multi-task MFR ⊕ dir-LP ⊕ structural | code | ⏳ pending (task #8) |
| T1/T2/T3 tables + notebook | code | ⏳ pending (task #9) |

## Budget decision (from the transfer sweep, 2026-07-09)

**Pretrain budget for E2–E4 = 40k episodes**, not 120k. The budget sweep
(`run_budget_sweep.sh`) showed downstream **classification is flat from 20k** and
**regression peaks ~40–60k then *degrades* toward 110k** — NM (instance
discrimination) collapses the continuous variation regression needs, so more NM
training actively hurts transfer. 40k is the regression peak and ~3× cheaper
(~1.5 hr vs ~4.6 hr per arm).

> ⚠️ **Off-by-one — use `epochs: 5`, not 4, for a true-40k checkpoint.** The trainer
> loops `trange(steps = epochs*dataset_len_cap)` over `e = 0 … steps-1` and only
> checkpoints at `e % 10000 == 0, e ≠ 0`. So `epochs: 4` (steps=40000) saves its
> **last checkpoint at 30000**, never 40000 — the same reason B0/E1 (`epochs:12`)
> top out at 110k, not 120k. **The first E2 run used `epochs:4` and therefore only
> reached a 30k final ckpt** (its run *completed* — "Training finished / best step
> 30000" — it was not killed). To land a clean `state_dict_40000.ckpt`, E-arm configs
> now use **`epochs: 5`** (steps=50000 → saves 10/20/30/**40**k; the ~10k steps past
> 40k are unused). `configs/{E2,E2b}.yaml` are set to `epochs: 5`; clone this for
> E3/E4. B0/B1/E1 already have real 40k ckpts from their 120k runs, so they need no
> rerun — the matched eval reads their `state_dict_40000.ckpt`.

**Matched-budget comparison:** the reading chain (E2−E1, E3−E2, …) must compare arms
at the SAME step, so evaluate E2–E4 (40k) against **B0/B1/E1 at their 40k checkpoints**
(`state_dict_40000.ckpt`), which already exist — not their 110k versions. Build E2–E4
configs with `epochs: 4`.

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

## Step 3 — Diagnostics (PRIMARY evidence; frozen encoders, no training)

Run after the eval sweep (they reuse `model_list.txt`). All offline apart from
GPU forward passes.

```bash
cd /dataMeR1/phil/gfm/prodigy
ML=scripts/experiments/topology_feature_ssl/model_list.txt

# T2 — 2x2 ablation: retained fraction under random-feat / rewired-edge / both.
# (intact reference = the Step-2 eval sweep; this runs the 3 corrupted conditions)
MODEL_LIST=$ML bash scripts/experiments/topology_feature_ssl/run_2x2_ablation.sh --gpus 0,1,2,3

# T3 — capability probes: linear-probe AUC on planted single-rule synthetic graphs.
python3 scripts/experiments/topology_feature_ssl/make_probe_graphs.py \
  --out-dir /dataMeR1/phil/data/synthetic_probes/graphs        # once
MODEL_LIST=$ML bash scripts/experiments/topology_feature_ssl/run_capability_probes.sh --gpus 0,1,2,3

# Leakage control: raw-structural-feature -> regression-target ceiling (no encoder).
python3 scripts/experiments/topology_feature_ssl/leakage_baseline.py --data-root /dataMeR1/phil/data
```

Outputs land in `scripts/plotting/topology_feature_ssl/data/`:
`ablation_2x2.csv` (T2), `capability_probes.csv` (T3), `leakage_baseline.csv`.
`parse_2x2.py` / `parse_capability_probes.py` also print the T2 / T3 tables.

## E-arm evals use a separate structural sweep

E1 uses the 3 degree-only structural inputs (input_dim 771), so its evals must
pass `--structural-features directed3` and use an E-arm-only model list — B0/B1
(768) and E1 (771) cannot share one sweep. Keep future E2-E4 configs and eval
commands on the same structural mode they were trained with.

```bash
# pretrain E1 (structural inputs; NM). One GPU.
tmux new-session -d -s tfssl_E1 'bash -lc "bash scripts/experiments/topology_feature_ssl/train_arm_tucker.sh E1 --device 2"'

# eval / diagnostics for the E-arms: E-only list + matching STRUCTURAL mode
STATE_DIR=/dataMeR1/phil/gfm/prodigy/state ARMS="E1 E2 E3 E4" \
  bash scripts/experiments/topology_feature_ssl/make_model_list.sh   # -> model_list.txt (E-arms)
STRUCTURAL=directed3 MODEL_LIST=scripts/experiments/topology_feature_ssl/model_list.txt \
  bash scripts/experiments/topology_feature_ssl/run_eval_sweep.sh --gpus 0,1,2,3
STRUCTURAL=directed3 MODEL_LIST=scripts/experiments/topology_feature_ssl/model_list.txt \
  bash scripts/experiments/topology_feature_ssl/run_2x2_ablation.sh --gpus 0,1,2,3
STRUCTURAL=directed3 MODEL_LIST=scripts/experiments/topology_feature_ssl/model_list.txt \
  bash scripts/experiments/topology_feature_ssl/run_capability_probes.sh --gpus 0,1,2,3
```

(Structural features are computed once per graph and cached as
`<graph>.structural_directed3.pt`, so the first E-arm eval on each graph pays the
feature computation cost and the rest reuse it.)

## Step 4 — E2 true-40k pretrain + matched-40k eval (completes E2's evaluation)

Prereqs verified present on Tucker (2026-07-10): merged corpus, `…structural_directed3.pt`
cache, and all 5 probe graphs. No rebuild needed. B0/B1/E1 already have real
`state_dict_40000.ckpt`; only E2 needs a rerun (its first run stopped at a 30k final
ckpt — see the off-by-one note above).

```bash
# --- 0. sync Tucker to the branch (laptop -> origin -> Tucker). Tucker was parked at
#        5679f12 with a dirty tree; discard the already-in-origin cruft, then pull.
#        These 6 files must be on Tucker: eval_ckpts_all_graph_tasks_tucker.py,
#        run_eval_sweep.sh, run_2x2_ablation.sh, run_capability_probes.sh,
#        configs/E2.yaml, configs/E2b.yaml.
cd /dataMeR1/phil/gfm/prodigy
git stash -u || true                 # or checkout/rm the already-committed M/?? files
git pull --ff-only                   # to the branch tip incl. the E2 40k / E2b changes
rm -rf state/tfssl_E2_smoke_*        # optional: keep `ls -dt tfssl_E2_*` unambiguous

# --- 1. pretrain E2 to a true 40k (epochs:5 -> final ckpt state_dict_40000). ~1.5h.
#        (GPUs 0-3 were free on 2026-07-10; nvidia-smi first.) run in tmux, login shell
#        so `conda activate` inside train_arm_tucker.sh works (detached-tmux conda gotcha).
tmux new-session -d -s tfssl_E2_40k \
  'bash -lc "bash scripts/experiments/topology_feature_ssl/train_arm_tucker.sh E2 --device 0"'
# watch: tmux capture-pane -pt tfssl_E2_40k | tail ; ls state/tfssl_E2_*/checkpoint/ | tail
# DONE when state_dict_40000.ckpt exists (and 50000 does NOT — 40k is the final ckpt).

# --- 2. matched-40k eval: finds E2's 40k ckpt, matches B0/B1/E1 at 40k, runs the full
#        6-target panel + static-LP + cls + 2x2 + probes + trivial + leakage, parses to
#        the *_40k CSVs, and renders RESULTS_matched40k.md. Needs conda active in-shell
#        (the script calls python3 directly), so source+activate at the top.
tmux new-session -d -s tfssl_m40k 'bash -lc "\
  source \$(conda info --base)/etc/profile.d/conda.sh && conda activate prodigy && \
  export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\${LD_LIBRARY_PATH:-} && \
  cd /dataMeR1/phil/gfm/prodigy && \
  GPUS=0,1,2,3 bash scripts/experiments/topology_feature_ssl/run_matched40k_tucker.sh \
  > /tmp/tfssl_m40k.log 2>&1"'
# watch: tail -f /tmp/tfssl_m40k.log ; DONE at line "MATCHED40K_DONE".
```

Lands (keyed `model = <arm>_40k`): `B0_40k/B1_40k/E1_40k/E2_40k` rows in
`scripts/plotting/{node_regression,static_link_prediction,node_classification}/data/*.csv`;
`ablation_2x2_40k.csv`, `capability_probes_40k.csv`, `trivial_baselines.csv`,
`leakage_baseline_6panel.csv` in `…/topology_feature_ssl/data/`; and
`RESULTS_matched40k.md`. Then commit the CSVs on Tucker + pull to the laptop (or scp the
5 CSVs + RESULTS_matched40k.md back) to update FINDINGS/RESULTS.

## Step 5 — E2b drop-BN encoder retry (the next fork)

E2b = E2 + `no_bn_encoder:true` (drop the conv-output BatchNorm that washes out
sum-aggregation's count magnitude — the named E2 culprit). Same 40k budget. Can run in
**parallel** with Step 4's E2 pretrain (different GPU). Eval MUST carry all three of
`STRUCTURAL=directed3 GNN_TYPE=sage_multi NO_BN_ENCODER=1`, or the BN-free state_dict
won't load.

```bash
# --- 1. pretrain E2b to 40k (epochs:5, no_bn_encoder). ~1.5h. Parallel with E2 (GPU 1).
tmux new-session -d -s tfssl_E2b \
  'bash -lc "bash scripts/experiments/topology_feature_ssl/train_arm_tucker.sh E2b --device 1"'

# --- 2. eval E2b vs the 40k family. Name it E2b_40k to sit beside E1_40k/E2_40k.
cd /dataMeR1/phil/gfm/prodigy
d=$(ls -dt state/tfssl_E2b_*/ | head -1)
echo "E2b_40k ${d}checkpoint/state_dict_40000.ckpt" \
  > scripts/experiments/topology_feature_ssl/model_list_E2b.txt
ML=scripts/experiments/topology_feature_ssl/model_list_E2b.txt
STRUCTURAL=directed3 GNN_TYPE=sage_multi NO_BN_ENCODER=1 MODEL_LIST=$ML \
  bash scripts/experiments/topology_feature_ssl/run_eval_sweep.sh --gpus 0,1,2,3
STRUCTURAL=directed3 GNN_TYPE=sage_multi NO_BN_ENCODER=1 MODEL_LIST=$ML \
  bash scripts/experiments/topology_feature_ssl/run_2x2_ablation.sh --gpus 0,1,2,3
STRUCTURAL=directed3 GNN_TYPE=sage_multi NO_BN_ENCODER=1 MODEL_LIST=$ML \
  bash scripts/experiments/topology_feature_ssl/run_capability_probes.sh --gpus 0,1,2,3
```

Read (vs E1_40k / E2_40k): **capability probes** (count / in-deg / out-deg) rising above
E2's ~0.6 cap ⇒ BN wash-out *was* the culprit and count became representable; probes
flat ~0.6 with reg/LP still ≤ E1 ⇒ multi-aggregation adds nothing and the encoder axis
is closed → pivot to the objective axis (E4). Add `E2b_40k` to `analyze_matched40k.py`
`ARMS` once its rows land to fold it into the matched table. (If drop-BN moves the
probes, the fast follow is a degree-scaler PNA aggregator — a new conv variant, not a
config flag.)
