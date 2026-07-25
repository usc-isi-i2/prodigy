# Findings — the mixed-objective lattice on valid metrics only

The corrected read of the 7-arm {NM, CL, FP} lattice after the 2026-07-23 LP-evaluator
rescore. **Supersedes the LP sections and all "emergent MIX link prediction" claims in
`multitask_ssl_rotation/FINDINGS.md` and `multitask_ssl_pairs/FINDINGS.md`.** Their
classification/regression sections remain valid and are consolidated here. Defect
details and the full 15-arm LP table (incl. the msc corpora arms) are in
[FINDINGS_rescore.md](FINDINGS_rescore.md).

## 1. Executive summary

**The 3-way-synergy story is dead.** On the old evaluator the joint bar
min(classification, LP) ran singles .42 → pairs .32 → triple .76 — a super-additive
leap only at k=3. On valid LP the same bar is **NM .757 > NMFP .738 > MIX/NMCL .680 >
CLFP/CL .543 > FP .492**: it *peaks at the single NM arm* and every objective added to
NM strictly lowers it. Link prediction is an NM main effect that rotation dilutes;
nothing emerges at k=3.

**What survives is a weaker, compromise-shaped generalist claim.** MIX is still the
only arm above chance on all three task families at once (cls .795, reg +.097, LP
.680 at +.036 over the best heuristic floor). But that is no longer an emergent
capability — it is diluted NM plus retained FP regression — and NM beats MIX outright
on two of the three tasks. "Rotation buys breadth nearly for free" is now "rotation
trades the top LP/cls arm for the only positive regression among structure-bearing
arms."

## 2. The lattice on valid numbers (1 seed, matched 40k compute)

Mean over datasets: classification = 10-shot ROC-AUC (2 datasets), regression =
Spearman (4 datasets), LP = pair ROC-AUC, degree-matched negatives (5 datasets);
LP floor margin vs best heuristic on the same pair set.

| k | arm | cls AUC | reg ρ | LP AUC | LP vs floor |
|---|---|---|---|---|---|
| 1 | NM | **.810** | −.001 | **.757** | **+.113** |
| 1 | CL | .638 | −.128 | .543 | −.101 |
| 1 | FP | .492 | **+.166** | .499 | −.145 |
| 2 | NMFP | .802 | −.098 | .738 | +.094 |
| 2 | NMCL | .800 | −.144 | .679 | +.035 |
| 2 | CLFP | .601 | +.110 | .543 | −.101 |
| 3 | MIX | .795 | +.097 | .680 | +.036 |

## 3. Reading

1. **LP is an NM main effect with monotone dilution.** Every NM-containing arm clears
   the heuristic floors (+.035 to +.113); every arm without NM sits below them
   (−.089 or worse). Within NM-containing arms the ordering follows NM's share of the
   rotation: NM (1/1) .757 > NMFP (1/2) .738 > NMCL (1/2) .679 ≈ MIX (1/3) .680.
   NM is the best arm on **all five** datasets individually, including the fully
   held-out twibot20 (.835 vs floor .726) — the transfer is real and out-of-domain.
2. **Classification is an NM property, unchanged.** Any NM-containing arm lands
   ~.80; CL .638, CLFP .601, FP at chance. Combination neither helps nor hurts here
   (.795–.810 band).
3. **Regression is the one place combination does something odd.** FP alone +.166,
   CLFP +.110, MIX +.097 — but both NM-pairs go *negative* (NMFP −.098, NMCL −.144)
   despite NM alone being flat (−.001). Taken literally, adding NM to FP destroys
   FP's regression signal in a pair yet MIX retains it. With 1 seed and 4 noisy
   datasets (per-dataset spread −.37 to +.29) this is the least trustworthy row of
   the table — treat as a hypothesis, not a finding.
4. **No arm is Pareto-dominant, but NM dominates the structural axis.** The old
   capability-plane picture (MIX alone in the top-right quadrant) inverts: NM now
   occupies the top-right on (cls × LP), and MIX's only remaining edge is the
   regression column. A joint-loss or reweighted rotation aimed at "keeping MIX's LP"
   has no target — the ceiling it would chase is NM itself.
5. **cp_hk stays at chance for every arm** (.484–.556), consistent with its isolation
   in the single-source transfer matrix — corpus composition, not objective choice,
   is the binding constraint there.

## 4. Caveats

- 1 seed throughout; eval episodes are split-seeded, so cross-arm contrasts are
  paired per dataset but no seed-CI exists. The LP ordering is consistent across all
  5 datasets (and across the mtr/msc corpora in FINDINGS_rescore.md), so the NM main
  effect is robust; the regression signs (reading 3) are not.
- Classification covers only 2 datasets (election2020 near ceiling for NM-arms;
  twibot20 does the discriminating).
- LP headline is degree-matched negatives; `hard_2hop` is punishing by construction
  and `random` inflates — both are in the data for sensitivity checks.
- Temporal LP has never been validly measured (same evaluator defect, unrescored).

## 5. Reproduce

```bash
/opt/homebrew/bin/python3.11 build_valid_dataset.py   # rebuild + sanity vs FINDINGS_rescore.md
```

Table 2 is `data/combined_valid.csv` pivoted on (model × task), metrics
{roc_auc, spearman, auc, margin_vs_floor}, sources mtr/mtp; per-dataset LP is in
`data/link_prediction_valid.csv`.
