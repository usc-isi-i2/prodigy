# mix_slp_ablation — is MIX's emergent static-LP topological or a feature artifact?

Eval-time 2×2 ablation of the headline finding of `multitask_ssl_rotation`
(see its `FINDINGS.md`): MIX (per-episode nm/cl/fp rotation) is the only arm
whose frozen embeddings support above-chance 0-shot static link prediction
(mean AUC 0.759 across 4 datasets), while NM/CL/FP sit at/below chance
(≤0.467). This experiment asks **why**: does MIX's sLP signal come from the
**true adjacency carried through message passing** (genuinely topological) or
from **feature homophily of the node bag** (a feature artifact)?

The answer **gates** the interpretation of the parallel training workstream:
if MIX's sLP survives edge rewiring, "emergent topological transfer" is a
feature artifact and must be re-interpreted.

## Design

Frozen 30k checkpoints, **no retraining**. Static-LP only (0-shot,
`--slp-n-query 4`, hard negatives), 2 arms × 4 conditions × 4 datasets:

- **Arms:** `MIX` (treatment) and `NM` (control; at-chance anchor).
- **Datasets:** `midterm`, `ukr_rus_twitter`, `covid19_twitter` (in-domain),
  `twibot20` (held-out).
- **Conditions** (eval-graph interventions, applied per sampled subgraph as
  eval-time augs — `data/augment.py`, composed by `--ablate_*` in
  `experiments/params.py`):

| condition | flags | run tag | what changes |
|---|---|---|---|
| none | — | `_slp_` | unmodified graph; must reproduce the FINDINGS anchor (MIX 0.759 / NM 0.467 mean) |
| rewire | `--ablate-edges rewire` | `_slp_ablE_` | message passing sees a **random same-size edge set** over the subgraph's node support; the LP task still scores the **true** held-out edges (positives/negatives from the original adjacency) |
| permute | `--ablate-features permute` | `_slp_ablP_` | feature rows **shuffled across nodes** within each subgraph (feature→node assignment randomized); true edges kept |
| both | both flags | `_slp_ablPE_` | rewire + permute |

Note on `rewire`: it is **matched-edge-count uniform rewiring** on the
subgraph's endpoint support (the existing, previously validated `AblateEdges`
transform from the tfssl 2×2), not a strictly degree-preserving
configuration-model rewire. It destroys adjacency *and* degree sequence within
the subgraph; for the gating question (feature artifact vs topology) this is
sufficient — a feature-homophily signal is untouched by any edge manipulation.

## Interpretation matrix

| signal source | rewire | permute | verdict pattern |
|---|---|---|---|
| **feature homophily** (endpoints score high because their *features* match; mean aggregation over the node bag) | **survives** | **dies** | feature artifact |
| **true topology in the embedding** (adjacency carried by message passing; e.g. neighbor-pool containment under mean aggregation) | **dies** | **survives** | genuinely topological |
| both channels | drops | drops | mixed |

Each cell is read against **chance = 0.50** and against the **none** anchor.
NM should sit at chance everywhere (a constant-predictor control). **The
gating verdict:** MIX surviving `rewire` ⇒ the emergent-sLP finding is a
feature artifact.

## Attribution / reproducibility

- Eval episodes are seeded by split name (`sum(ord(split))`), **not** by
  `--seed`, so episode content (centers, positives, negatives) is identical
  across all 4 conditions; any AUC delta is attributable to the ablation.
- The ablation RNG (torch `randperm`/`randint` in the augs) derives from
  `--seed 0` (`run_single_experiment.py` seeds torch/np/random), fixed via
  `SEED=0` in `run_2x2_slp.sh`.

## How to run (Tucker)

```bash
cd /dataMeR1/phil/gfm/prodigy-abl   # worktree on exp/mix-slp-ablation
STATE_DIR=/dataMeR1/phil/gfm/prodigy-mtr/state \
  bash scripts/experiments/setup/mix_slp_ablation/make_model_list.sh   # pins 30k ckpts

# sanity first: none-condition, MIX only, one dataset — must land near 0.755
# (covid19) before the ablations are worth running. See EXECUTION.md.

bash scripts/experiments/setup/mix_slp_ablation/run_2x2_slp.sh --gpus 0
```

Results: `run_2x2_slp.sh` ends by writing
`scripts/experiments/analysis/mix_slp_ablation/data/slp_ablation_2x2.csv`
via `parse_slp_2x2.py` (reads each run's `data/metrics_test.json` directly;
the shared `parse_benchmark_eval_logs.py` SLP regex does not match `_abl*`
tags). Findings: `scripts/experiments/analysis/mix_slp_ablation/FINDINGS.md`.
