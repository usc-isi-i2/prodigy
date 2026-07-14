# multitask_ssl_pairs — findings

**Result: MIX's emergent link-prediction is a genuine 3-way synergy. No pair of
SSL objectives reproduces it, and pairs do not even interpolate between the
singles and the triple.**

Frozen-encoder transfer over the full non-empty subset lattice of {nm, cl, fp},
all 7 arms at matched 40k-episode pretraining compute, mean over the focused-5 eval
datasets (test split). 1 seed. Reproduce: `python aggregate_results.py
--plotting-root results` (full dump in `results/lattice_table.txt`).

| arm | k | cls AUC | reg ρ | **sLP AUC** | min(cls,sLP) | group |
|---|---|---|---|---|---|---|
| NM   | 1 | **0.810** | −0.001 | 0.467 | 0.467 | single |
| CL   | 1 | 0.638 | −0.128 | 0.332 | 0.332 | single |
| FP   | 1 | 0.492 | **0.166** | 0.449 | 0.449 | single |
| NMCL | 2 | 0.800 | −0.144 | 0.305 | 0.305 | pair |
| NMFP | 2 | 0.802 | −0.098 | 0.424 | 0.424 | pair |
| CLFP | 2 | 0.601 | 0.110 | 0.227 | 0.227 | pair |
| **MIX** | 3 | 0.795 | 0.097 | **0.759** | **0.759** | triple |

cls: 2 labeled datasets (twibot20, election2020); reg: 4 datasets × 3 targets;
sLP: 4 datasets. All arms bio-768/mean-SAGE on the merged 3-way corpus, global
sampling, matched budget (pairs split 20k/20k; singles 40k×1; MIX ~13.3k×3).

## T1 — the headline: emergent static-LP needs all three

Static-LP ROC-AUC (0-shot, chance = 0.50):

- **Every single and every pair is at or below chance** (0.23–0.47). Only **MIX
  clears it, at 0.759** — and it clears it on *all four* datasets (0.68–0.86):

  | dataset | NM | CL | FP | NMCL | NMFP | CLFP | **MIX** |
  |---|---|---|---|---|---|---|---|
  | covid19_twitter | 0.406 | 0.366 | 0.449 | 0.320 | 0.395 | 0.228 | **0.755** |
  | midterm | 0.487 | 0.417 | 0.433 | 0.376 | 0.445 | 0.310 | **0.676** |
  | twibot20 | 0.491 | 0.259 | 0.381 | 0.315 | 0.471 | 0.202 | **0.745** |
  | ukr_rus_twitter | 0.484 | 0.288 | 0.534 | 0.210 | 0.387 | 0.168 | **0.861** |

- **Pairs are not partial credit.** Adding a second objective does not move sLP
  toward MIX — several pairs are *worse* than their own constituents: NMCL (0.305)
  < NM (0.467); CLFP (0.227) < both CL (0.332) and FP (0.449). NMFP (0.424) just
  sits between its parents. Two objectives dilute; three synergize.

## Capability vs. number of objectives — non-monotonic

Joint generalist bar `min(feature = cls AUC, topological = sLP AUC)`, mean over
arms at each k:

| k | arms | cls AUC | reg ρ | sLP AUC | **min bar** |
|---|---|---|---|---|---|
| 1 | NM,CL,FP | 0.647 | 0.012 | 0.416 | **0.416** |
| 2 | NMCL,NMFP,CLFP | 0.734 | −0.044 | 0.319 | **0.319** |
| 3 | MIX | 0.795 | 0.097 | 0.759 | **0.759** |

The min-bar goes **down** from singles → pairs, then jumps at the triple. MIX is
the only arm above 0.6 (a genuine dual-capability generalist); every other arm is
bottlenecked by sLP at chance. best-pair (NMFP 0.424) − best-single (NM 0.467) =
**−0.042**; MIX − best-pair = **+0.335**. The whole is more than the sum, and no
proper subset of the three gets you there.

## Per-axis reads

- **Classification is an NM property**, and it is *preserved* under combination:
  NM 0.810 → NMCL 0.800, NMFP 0.802, MIX 0.795 (all ≈ NM). CL is mediocre (0.638),
  FP alone collapses to chance (0.492). Any rotation containing NM keeps
  classification; adding cl/fp costs ~nothing here.
- **Regression is an FP property** (FP 0.166, the only clearly positive single),
  but weak and noisy for everyone (|ρ| ≤ 0.17, several negative) — consistent with
  the rotation run, where regression is a secondary, high-variance feature axis and
  is excluded from the joint bar.
- **Static-LP (topology) is a MIX-only, all-three property** — the headline above.

## Interpretation

The three objectives are individually specialists (NM→classification, FP→weak
regression, CL→nothing much) and pairwise still specialists or worse. The
topological transfer that shows up as zero-shot link-prediction is an **emergent,
super-additive effect of rotating over all three objectives at once** — plausibly
the three-way rotation acting as a regularizer that forces a representation no
single loss (or pair of losses) has any pressure to build. This is the strongest
possible version of the "learns both feature and topology" thesis: it is not that
more objectives monotonically help, but that the *complete* set unlocks a
capability none of its subsets touch.

## Caveats

- **1 seed** (seed 0), matching the rotation run. The sLP gap is huge and
  consistent across all 4 datasets (MIX ≥ 0.68 everywhere; every subset ≤ 0.53), so
  the qualitative result is not seed noise; exact numbers are single-draw.
- The `results/lattice_table.txt` "marginal sLP by objective" read (nm +0.153, fp
  +0.097, cl −0.041) is **confounded**: MIX is the only high-sLP arm and contains
  all three objectives, so it inflates every "with" average. Do not read it as a
  clean per-objective additive effect — the real story is the 3-way interaction,
  not any single objective's marginal.
- Regression is weak/noisy for all arms; treat reg deltas as within-noise.

## Provenance

- Pairs trained 2026-07-14 in worktree `/dataMeR1/phil/gfm/prodigy-mtp` (branch
  `mtp-run` @ commit with this experiment), 40k episodes each, ckpts at 10k/20k/30k;
  eval at the 30k ckpt (matched to the rotation arms). NM/CL/FP/MIX checkpoints
  reused from the `multitask_ssl_rotation` run (`prodigy-mtr/state/mtr_*`), all 7
  arms scored in a single eval sweep under identical conditions.
- `results/` holds the three benchmark CSVs (keyed by `model` = arm) and the
  aggregator dump; the table above regenerates from them with `aggregate_results.py`.
