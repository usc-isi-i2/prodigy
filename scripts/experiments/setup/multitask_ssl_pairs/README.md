# multitask_ssl_pairs

**Question.** We trained one encoder on each SSL objective alone (NM, CL, FP) and
on a 3-way per-episode rotation (MIX). MIX was the only arm that became a
*generalist* — near-best classification **and** the only arm with real static-LP
(emergent topological transfer none of nm/cl/fp achieve alone). **Which
combination is responsible?** Does any *pair* of objectives already unlock it, or
is all three necessary — and is one objective (nm? fp?) the driver?

This experiment fills the middle rung of the lattice: the three **pairs**.

## Design — the subset lattice of {nm, cl, fp}

| k | arms | how trained |
|---|---|---|
| 1 (single) | NM, CL, FP | `multitask_ssl_rotation` controls (one objective, 40k episodes) |
| **2 (pair)** | **NMCL, NMFP, CLFP** | **this experiment (1:1 rotation, 40k total)** |
| 3 (triple) | MIX | `multitask_ssl_rotation` treatment (1:1:1 rotation, 40k total) |

7 non-empty subsets → 7 arms, all at **matched 40k-episode pretraining compute**:
singles spend all 40k on one objective, pairs split ~20k/20k, MIX splits ~13.3k×3.
The *only* thing that varies is which objectives share the budget — so any change
in downstream capability is attributable to objective composition, not compute.

## Config-only: zero code changes

Each pair reuses the **exact** `nm_fp_cl` rotation machinery with one objective's
count zeroed. `MultiTaskSplitBatch` builds its per-episode schedule as
`[i for i,c in enumerate(counts) for _ in range(c)]`, so a `0` count means that
task is **never sampled**; and the FP reconstruction loss only fires on episodes
the Collator tags `graph.mix_is_fp`. So a pair is byte-for-byte the same code path
as NM/CL/FP/MIX — just a different `mix_task_counts`:

| arm | `task_name` | `mix_task_counts` (nm,cl,fp) | active objectives |
|---|---|---|---|
| NMCL | `nm_fp_cl` | `1,1,0` | nm + cl (both metric loss) |
| NMFP | `nm_fp_cl` | `1,0,1` | nm (metric) + fp (recon) |
| CLFP | `nm_fp_cl` | `0,1,1` | cl (metric) + fp (recon) |

(For NMCL, an unused `aux_header` is built but receives no gradient — inert.)
Everything else — merged 3-way corpus, bio-768/mean-SAGE, global sampling,
`n_way=30`, `batch_size=1`, aug knobs (cl=NZ0.2, fp=0.3) — is copied from
`multitask_ssl_rotation/configs/MIX.yaml`, so pairs are exact sub-combinations.

## Evaluation

Identical to the rotation sweep, so all 7 arms land in one table: freeze each
encoder and run the joint benchmark across the focused-5 datasets (3 in-domain:
ukr_rus, covid, midterm; 2 held-out: twibot20, election2020):
- **node classification** (10-shot) — feature axis
- **static link prediction** (0-shot, n_query 4) — topological axis (the headline)
- **node regression** (10-shot, log1p) — secondary, noisy feature axis

`aggregate_results.py` merges the pairs with the NM/CL/FP/MIX rows into the
subset-lattice table + the joint generalist bar `min(cls AUC, sLP AUC)` + the
"marginal sLP by objective" read (mean sLP over arms that do vs don't rotate each
objective) that pins the LP driver.

## Hypotheses / reads

- **If a pair already clears static-LP** (AUC > ~0.6 while all singles are at
  chance): LP emergence needs only 2 objectives — read off *which* pair to name
  the driver (e.g. only NMFP → nm×fp interaction; only fp-containing pairs → fp
  drives it).
- **If no pair clears it and only MIX does**: all three are jointly necessary — a
  genuine 3-way synergy.
- **Monotonicity**: does `min`-bar rise with k (single < pair < triple), or does
  one pair already match MIX (diminishing returns)?

See `EXECUTION.md` for commands and `FINDINGS.md` (written after the sweep) for the
result.

## Archived Tucker artifacts

`model_list_30k_archived.txt` preserves the exact pair and rotation checkpoint
paths used by the completed sweep. The retired worktrees' ignored files are in
`/dataMeR1/phil/gfm/artifacts/worktree_cleanup_20260805/{mtp,mtr}-ignored.tar`.
The full pre-cleanup branch snapshot is tagged
`archive/preservation-multitask-ssl-pairs-2026-08-05`.
