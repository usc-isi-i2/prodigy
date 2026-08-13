# NM ladder — order robustness (multiple graph orders)

**Status:** configs + scripts ready, 2026-07-23. Nothing trained yet — the gate runs first.

## What we are doing

The published NM interpolation ladder (`nm_ladder_fillin`) added the 8 source graphs in
**one fixed order** (the table-column order). Its headline result — *a test graph's AUC
stays flat at its zero-shot transfer level until its graph enters the training merge, then
jumps and holds* — is therefore measured on a single curriculum. This experiment re-runs
the ladder under **two additional graph orders** so the staircase can be claimed as a
property of *set membership*, not of the particular sequence we happened to pick.

We vary **order only**. Training seed is held fixed at 0 (deliberate: order is the
confound; seed noise is already known to be sub-.02 on this metric).

## Claim being defended

> Entering the merge causes the jump; where in the sequence it enters does not.

Operationally: for every (graph, order) pair, the entry-aligned Δ (AUC just after its
graph enters − AUC just before) is positive, and each graph's post-entry level is
approximately independent of entry position.

## The three orders

Ordered by **donor strength** = mean off-diagonal transfer as a source, computed from
`analysis/transfer/matrices/prodigy_nm/single_source/nm_single_source_matrix/data/nm_single_source_matrix.csv`:

| ukraine | covid | twibot20 | midterm | ukraine-susp | hongkong | covid-political | election2020 |
|--------:|------:|---------:|--------:|-------------:|---------:|----------------:|-------------:|
| .849 | .847 | .818 | .778 | .739 | .695 | .662 | .649 |

| | Order | Rationale |
|---|---|---|
| **A** | ukraine → covid → midterm → covid-political → election2020 → ukraine-susp → twibot20 → hongkong | Published topical order. Already run. |
| **B** | covid → ukraine → twibot20 → midterm → ukraine-susp → hongkong → covid-political → election2020 | Donor strength **descending**. Prior coverage is maximal at every rung, so every later Δ is **minimised** — the worst case for our claim. |
| **C** | election2020 → covid-political → hongkong → ukraine-susp → midterm → twibot20 → ukraine → covid | Exact **reverse of B**. Weak/isolated graphs first — **maximises** Δs and stresses small-graph dilution under `balanced` sampling. |

B and C are exact mirrors, so one sentence defines both, and together they bracket the
mechanism already established in `nm_ladder_fillin` (Δ at entry ≈ own ceiling − best donor
already present). ukraine and covid are tied within noise (.849 vs .847); B starts with
covid so ukraine is not pinned to position 1 in two separate orders.

## Cost: 11 new training runs

| Order | Rungs to train | Reused |
|---|---|---|
| A | — | all 8 rungs exist (`nm_ladder_fillin`) |
| B | rungs 3–7 (**5 new**) | r1 `{covid}` = existing specialist row; r2 `{covid,ukraine}` = same *set* as A's rung 2; r8 = all8 |
| C | rungs 2–7 (**6 new**) | r1 `{election2020}` = existing specialist row; r8 = all8 |

Rung 8 is order-invariant (always all8) and rung 1 of any order is that graph's
single-source specialist row from `nm_single_source_matrix`. Verify config parity once
before reusing a specialist row as a rung 1 — they should match (single-source ukraine
.947 vs ladder rung 1 .948).

Plus 11 × 8 = 88 NM eval jobs.

## Protocol

Inherits `nm_ladder_fillin` unchanged: 256·S,U,M base, no aug, `attr_regression_weight=0`,
within-balanced episodes (`neighbor_sampling_episode_source: graph_id` + `balanced`),
`epochs:5`/`checkpoint_step:10000` so each run self-terminates with `state_dict_40000`,
eval via `eval/eval_ckpts_all_graph_tasks_tucker.py` at NM 30-way / 3-shot / matched-40k.

**One code change, to avoid rebuilding merged graphs.** Naively each order needs its own 7
nested merges (~100 GB each). Instead we add a source-subset knob to the episode sampler
(alongside the existing `neighbor_sampling_episode_source`) and train **every rung of every
order from the single existing all8 graph**, restricting which `graph_id`s episodes may be
drawn from. The merge is disjoint (no cross-source edges) and episodes are already confined
to one `graph_id`, so restricting the allowed source set is equivalent to training on that
sub-merge. **Gate:** reproduce A's rung 4 via the subset knob and check it matches the
published row within ~.01 before running anything else.

## Run order (Tucker)

Configs are generated, not hand-written — `make_configs.py` decides new-vs-reuse by
source set and writes `manifest.csv` alongside the 11 `train_ord*.yaml` files.

```bash
cd scripts/experiments/setup/nm_ladder_order_robustness-jul_23

# 0. regenerate configs / preview the plan (idempotent)
python3 make_configs.py --dry-run

# 1. THE GATE — train order A rung 4 through the subset knob (~80 min, 1 GPU)
tmux new-session -d -s nmlor_gate \
  'export PATH="/home/mhchu/miniconda3/bin:$PATH"; \
   GATE=1 GPUS="0" bash scripts/experiments/setup/nm_ladder_order_robustness-jul_23/run_all_train_tucker.sh \
   > scripts/experiments/setup/nm_ladder_order_robustness-jul_23/run_logs/gate.log 2>&1'

# 2. eval the gate on all 8 graphs, then check it against the published rung 4
GATE=1 STATE_DIR=/dataMeR1/phil/gfm/prodigy/state ./make_model_list.sh
GATE=1 GPUS="0,1,2,3" ./eval_ladder_tucker.sh
python3 check_gate.py --log-root /dataMeR1/phil/gfm/prodigy/log
#    exit 0 = PASS -> continue.  exit 1 = FAIL -> STOP, the shortcut is invalid.
#    exit 2 = incomplete (missing eval columns).

# 3. only if the gate PASSES: train the 11 rungs
#    ~80 min each at ~8.5 it/s (40k steps); 3 GPUs => 4 rounds => ~5.5h
tmux new-session -d -s nmlor_rungs \
  'export PATH="/home/mhchu/miniconda3/bin:$PATH"; \
   GPUS="0 1 2" bash scripts/experiments/setup/nm_ladder_order_robustness-jul_23/run_all_train_tucker.sh \
   > scripts/experiments/setup/nm_ladder_order_robustness-jul_23/run_logs/orchestrator.log 2>&1'

# 4. eval all 11 rungs on all 8 graphs (88 NM jobs)
STATE_DIR=/dataMeR1/phil/gfm/prodigy/state ./make_model_list.sh
GPUS="0,1,2" ./eval_ladder_tucker.sh
```

**GPU etiquette (2026-07-23):** GPUs 0–3 are ours on paper, but labmates `rdorn`/`ashreyas`
run a long-lived `ollama` server that holds a 520 MiB context stub on GPU 0 and the actual
model (~42 GB) on GPU 3, where it bursts to 75–95% utilization while serving. Use **GPUs
0, 1, 2** and leave GPU 3 alone. Our NM jobs draw ~2.5 GB at ~9% utilization, so sharing
GPU 0 costs them nothing; landing on GPU 3 would contend with live inference.

`DRY_RUN=1` previews any launch script without touching a GPU. See AGENTS.md for the
conda-on-PATH gotcha that makes detached tmux jobs die immediately.

The gate is a real stop, not a formality: if it fails, the 11 rungs are not merely noisy,
they are measuring something other than the sub-merges they claim to. The fallback is
building nested merges per order (~7 × ~100 GB per order), which is a different budget
conversation.

## Analysis / deliverable

Entry-aligned ("event study") aggregation: re-index each test graph's curve by rungs
relative to **its own entry** (−2, −1, entry, +1, …) and overlay the three orders.
Headline statistic = sign test over the 8 graphs × 3 orders = 24 entry events.

With n=3 orders, do **not** report mean ± CI. Report per-graph entry-aligned curves for all
three orders plus min/max range, and cluster on order (one trained model produces 8 columns,
so columns within a rung are not independent).

## Known limitation (state it, don't hide it)

midterm and ukraine-suspended are mid-ranked donors, so they enter at positions 3–5 in all
three orders — their Δ is never measured against an empty or nearly-full merge. The defense
is that ordinal position is not the operative variable, prefix *composition* is: in A,
midterm enters a merge of `{ukraine, covid}` (two universal donors); in C it enters
`{election2020, covid-political, hongkong, ukraine-susp}` (four weak ones). Opposite
regimes, similar positions. Report composition alongside rung index.

## Related

- `setup/nm_ladder_fillin/` — the single-order ladder this extends (protocol source of truth).
- `setup/nm_single_source_matrix/` — donor strengths, and the specialist rows reused as rung 1.
- `analysis/transfer/ladders/prodigy_nm/canonical/nm_ladder/RESULTS.md` — the published 8×8 table (order A).
