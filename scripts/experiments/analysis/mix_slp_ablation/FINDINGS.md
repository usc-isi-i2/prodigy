# Findings — mix_slp_ablation (eval-time 2×2: rewire × permute)

**Question.** Is MIX's emergent 0-shot static-link-prediction ability
(`multitask_ssl_rotation` FINDINGS: mean AUC 0.759 while NM/CL/FP sit at/below
chance) genuinely **topological** — true adjacency carried through message
passing — or a **feature artifact** (feature homophily of the node bag)?
This gates the interpretation of the parallel training workstream.

**Design** (see `scripts/experiments/setup/mix_slp_ablation/README.md`):
frozen 30k checkpoints (MIX treatment, NM at-chance control), static-LP only,
0-shot, 4 datasets, 4 eval-graph conditions applied per sampled subgraph at
eval time (no retraining): **none**, **rewire** (`--ablate-edges rewire`,
message passing sees a random same-size edge set over the subgraph's node
support while the task still scores the TRUE held-out edges), **permute**
(`--ablate-features permute`, feature rows shuffled across nodes; true edges
kept), **both**. Seed 0; eval episodes are seeded by split name, so episode
content is identical across conditions — deltas are attributable to the
ablation alone.

**Sanity anchor.** The `none` column reproduces the published numbers
**exactly** (e.g. MIX/covid19 0.75508 in both the original 2026-07-09 sweep
and this rerun) — same checkpoints, same split-seeded episodes, same flags.

## Results — test ROC-AUC (chance = 0.50)

Raw rows: [`data/slp_ablation_2x2.csv`](data/slp_ablation_2x2.csv)
(32 cells = 2 arms × 4 conditions × 4 datasets; regenerate the tables with
`python3 aggregate_2x2.py`).

### MIX (treatment)

| dataset | none | rewire | permute | both |
|---|---|---|---|---|
| covid19 (in-domain)     | 0.755 | 0.533 | 0.758 | 0.539 |
| midterm (in-domain)     | 0.676 | 0.561 | 0.674 | 0.546 |
| ukr_rus (in-domain)     | 0.861 | 0.644 | 0.868 | 0.677 |
| twibot20 (held-out)     | 0.745 | 0.565 | 0.748 | 0.569 |
| **mean**                | **0.759** | **0.576** | **0.762** | **0.583** |

### NM (control)

| dataset | none | rewire | permute | both |
|---|---|---|---|---|
| covid19     | 0.406 | 0.395 | 0.425 | 0.410 |
| midterm     | 0.487 | 0.487 | 0.494 | 0.478 |
| ukr_rus     | 0.484 | 0.479 | 0.555 | 0.525 |
| twibot20    | 0.491 | 0.519 | 0.498 | 0.485 |
| **mean**    | **0.467** | **0.470** | **0.493** | **0.474** |

### Interpretation matrix (pre-registered in the setup README)

| signal source | rewire | permute | observed for MIX? |
|---|---|---|---|
| feature homophily (feature artifact) | survives | dies | **no** — permute changes nothing (0.762 vs 0.759) |
| true topology in the embedding | dies | survives | **yes** — rewire collapses −0.18 toward chance; permute is a no-op |
| both channels | drops | drops | no |

## VERDICT

**MIX's emergent static-LP signal is genuinely topological, not a feature
artifact.** Destroying the eval graph's adjacency while keeping every node
feature (rewire) collapses MIX from 0.759 to 0.576 mean AUC (−0.18, on all
4/4 datasets); destroying the feature→node assignment while keeping the true
adjacency (permute) leaves MIX completely unchanged (0.762 vs 0.759; per-
dataset |Δ| ≤ 0.007). The double ablation tracks rewire (0.583), i.e. once
edges are gone there is no residual feature channel to remove — the two
manipulations are not redundant destroyers of one shared signal. The NM
control stays at chance in every condition (0.467–0.493), confirming the
ablations do not manufacture signal. This is the pattern predicted by the
"rotation taught adjacency" hypothesis and rules out the feature-homophily
artifact: **the gating condition for the parallel training workstream — "MIX
survives edge rewiring" — did NOT occur.**

Two caveats on the residual. (1) Under rewire MIX retains a small
above-chance residue (0.533–0.644, mean 0.576, largest on ukr_rus). The
rewire transform is uniform over the subgraph's endpoint support — it
destroys adjacency *and* within-subgraph degree (it is not a
configuration-model, degree-preserving rewire, a deliberate reuse of the
validated tfssl transform; noted as a deviation from the original
degree-preserving spec). What it does **not** touch is (a) the subgraph's
node-bag *composition*, which was neighbor-sampled from the TRUE graph before
the transform and remains a weak topological cue (endpoint bags of a true
edge overlap), and (b) the pooling-supernode scaffolding. The residue is
therefore best read as leftover topology in the bag composition, not as a
feature channel — consistent with it also surviving the double ablation.
(2) Single seed, but the effect is large, one-directional, and consistent
across all four datasets including held-out twibot20.

## Provenance

- Checkpoints: `mtr_{MIX,NM}_09_07_2026_15_09_52/checkpoint/state_dict_30000.ckpt`
  under `/dataMeR1/phil/gfm/prodigy-mtr/state/` (the FINDINGS checkpoints,
  pinned by step — not mtime).
- Run: Tucker worktree `/dataMeR1/phil/gfm/prodigy-abl`, branch
  `exp/mix-slp-ablation`, 2026-07-21, GPUs 0–3, ~35 min wall.
- Metrics were read from each run dir's `data/metrics_test_step0.json`
  (step-suffixed; the shared `parse_benchmark_eval_logs.py` SLP regex does
  not match `_ablE/_ablP/_ablPE` run dirs) by
  `scripts/experiments/setup/mix_slp_ablation/parse_slp_2x2.py`.
- The 09:39 sanity run of MIX/covid19/none was superseded in the CSV by the
  12:23 grid rerun of the same cell (latest-timestamp-wins; both scored
  0.75508, bit-identical).
