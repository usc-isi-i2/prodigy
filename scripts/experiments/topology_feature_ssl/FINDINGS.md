# topology_feature_ssl — Findings (B0 / B1 / E1)

**Date:** 2026-07-09. **Scope:** first 3 of the 6 planned arms + full diagnostics.
E2–E4 deferred. See `README.md` (plan), `RESULTS.md` (auto-rendered tables),
`EXECUTION.md` (ops), and the notebook `scripts/plotting/topology_feature_ssl/`.

## Setup
3-way merged retweet graph (**34M nodes**; ukr_rus 10.4M / covid 23M / midterm 342k),
within-source **balanced** NM episodes, **1 seed**, 120k episodes (final checkpoint
110k). Frozen-encoder eval on 5 datasets (covid/ukr/midterm in-domain + twibot20/
election2020 held-out). Arms — each **one lever off B0**:

- **B0** — control NM (mean-agg SAGE, bio features, undirected MP).
- **B1** — B0 + `NR0.3` (random-feature corruption of 30% of nodes).
- **E1** — B0 + `directed3` structural inputs `[in_deg, out_deg, log_deg]` (input_dim 771).

## Headline
**E1 (structural inputs) is the only lever that helps. B1 (cheap `NR` augmentation)
backfires. B0/NM is confirmed features-only and structurally blind.**

## Findings

1. **NM is features-only & structurally blind.** 2×2 ablation: rewiring the edges
   leaves feature-task performance unchanged (**retain 1.11**); capability probes
   sit at **chance (0.50–0.51) on every primitive** — the mean-agg encoder cannot
   represent count / degree / existence / conjunction. The plan's premise holds.

2. **Cheap augmentation (B1) is refuted — and actively harmful.** `NR0.3` did not
   force topology use: regression went **negative** (−0.11 content / −0.10 struct),
   static-LP **collapsed below chance** (0.36 — the corruption inverted the LP
   geometry), probes stayed at chance. The **free preview agrees**: fp (≈E3's
   masked-feature objective) does **not** beat NM on regression (mean fp−nm = −0.016).
   → Both cheap levers (data-side B1, fp objective) are null.

3. **Structural inputs (E1) are a partial win.** Best of the three on the **joint
   benchmark**: regression 0.13 (content) / 0.14 (struct), static-LP **0.76**, cls
   flat (~0.77). Makes **count + in-degree representable** (probes 0.64 / 0.59 vs
   chance). *But* structure-target regression only **matches the shot-matched
   leakage ceiling** (Δ ≈ −0.02 → passthrough, not learned), and the probes are only
   ~0.6 — **the mean-pool encoder dilutes even its own injected features.**

4. **NM anti-scales on regression.** Transfer budget sweep (20/40/60/110k):
   classification flat from 20k; regression **peaks ~40–60k then *degrades* to 110k**
   (instance discrimination collapses the continuous variation regression needs).
   → **E2–E4 budget = 40k** (`epochs: 4`), compared against B0/B1/E1 at their **40k**
   checkpoints (matched, and the regression peak).

## Reading-chain conclusions
- **B1 − B0 (augmentation lever):** the feature shortcut is **not removable by `NR`
  corruption**; the architecture/objective must change. Positive support for the
  E-arms, not a null.
- **E1 − B0 (structural input → representable):** E1 makes topology **representable
  and benchmark-useful**, but a count-blind mean encoder **can't fully *use* it**
  (at-passthrough structure regression, ~0.6 probes). The binding gap is
  **representational fit → E2** (count-capable / PNA aggregation).

## Methodological notes (things that changed a verdict)
- **Shot-matching the leakage baseline mattered.** A full-data Ridge ceiling (163k
  nodes) vs a 10-shot eval wrongly read E1 as passthrough on account_age; the
  shot-matched ceiling (0.06) flipped it — E1 (0.19) genuinely **beats passthrough
  on the content target**. Always compare the leakage probe at the eval's shot count.
- **The 2×2 on feature tasks doesn't discriminate topology-use.** reg/pl don't need
  edges, so rewired-edge ≈ 1.0 for every arm; it confirms B0 is features-only but
  E1's topology evidence lives in **static-LP + probes**, not the feature-task 2×2.
  Consider folding a topology task into T2's retained fraction next time.
- **Single seed; small deltas** (the May aug delta was +0.008). Treat T1 as
  confirmatory, T2/T3 as primary. NM 30-way test acc has ±0.12 std.

## Next
**E2 — expressive directed aggregator** (`aggr` mean→sum→PNA, in/out neighbors split,
readout mean→mean⊕sum⊕max) at 40k. Tests whether making structure *usable* (not just
representable) closes E1's passthrough gap and widens the probe / static-LP margins —
i.e. whether the binding constraint really is representational fit.
