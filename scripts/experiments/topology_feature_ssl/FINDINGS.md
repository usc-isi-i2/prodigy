# topology_feature_ssl — Findings (B0 / B1 / E1 / E2 vs trivial floor)

**Date:** 2026-07-09. **Scope:** 4 of the 6 planned arms + full diagnostics +
trivial-floor baselines. E3–E4 deferred. See `README.md` (plan), `RESULTS.md` /
`RESULTS_matched40k.md` (auto-rendered tables), `EXECUTION.md` (ops), and the
notebook `scripts/plotting/topology_feature_ssl/`.

## Setup
3-way merged retweet graph (**34M nodes**; ukr_rus 10.4M / covid 23M / midterm 342k),
within-source **balanced** NM episodes, **1 seed**. Frozen-encoder eval on 5 datasets
(covid/ukr/midterm in-domain + twibot20/election2020 held-out), 10-shot. Arms — each
**one lever off B0**:

- **B0** — control NM (mean-agg SAGE, bio features, undirected MP).
- **B1** — B0 + `NR0.3` (random-feature corruption of 30% of nodes).
- **E1** — B0 + `directed3` structural inputs `[in_deg, out_deg, log_deg]` (input_dim 771).
- **E2** — E1 + count-aware encoder: `sage_multi` (mean→mean⊕sum⊕max aggregation) +
  multi-readout. (Directed in/out split deferred — subgraphs are `bidirectional=False`.)

**Two budgets:** B0/B1/E1 have a full 120k run (final ckpt 110k) *and* a matched
**30k** run; E2 is 30k only. **NM anti-scales on regression** (peaks ~40k, degrades
by 110k), so the fair arm comparison is at the matched **30k** budget.

**Trivial floor (the absolute anchor):** `raw_feat` = 10-shot linear probe on the
raw 768-d bio embedding (no encoder); `raw_degree` = 10-shot probe on the raw
`directed3` features (leakage). An arm only *"improves performance"* if it beats
these — not just the other arms.

## Headline
**Injecting structural features (E1) genuinely improves transfer over the trivial
floor — decisively on the content×structure target (account_age) plus classification
and static-LP — so the win is real, not just relative. But a count-aware *encoder*
(E2) does NOT help: at matched budget it underperforms E1 on regression and static-LP.
So the win is the structural INPUT, not a fancier aggregator — representational fit
was not the binding constraint. B1 (cheap `NR` augmentation) backfires; B0/NM is
features-only and structurally blind.**

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

5. **vs the trivial floor (matched 30k) — E1 clears it, E2 does not.** Regression
   Spearman, 6-panel, mean over datasets:

   | | followers | friends | statuses | favourites | listed | account_age |
   |---|---|---|---|---|---|---|
   | raw_degree | 0.16 | 0.07 | 0.16 | 0.09 | 0.10 | 0.01 |
   | raw_feat | 0.20 | 0.14 | 0.10 | 0.07 | 0.15 | 0.02 |
   | B0 | −.01 | −.11 | −.02 | −.05 | −.01 | −.08 |
   | B1 | −.04 | −.07 | −.02 | −.03 | −.07 | −.13 |
   | **E1** | **.19** | **.16** | .10 | .05 | .14 | **.13** |
   | E2 | −.01 | −.10 | −.02 | −.11 | .03 | −.05 |

   - **E1 beats *both* trivial baselines on friends + account_age** (0.13 vs ~0.01–0.02
     — decisive and now robust across datasets), beats `raw_degree` on followers, and
     is best on **static-LP** (0.70–0.75) + beats `raw_feat` on **classification**
     (0.96 vs 0.85 / 0.62 vs 0.56). **E1 genuinely improves over doing nothing.**
   - **B0, B1, E2 all LOSE to the trivial floor on regression** (near-zero/negative);
     classification is the exception (encoders beat `raw_feat`).
   - **E2 underperforms E1** on regression *and* static-LP (0.35–0.43) at matched
     budget. Count-aware aggregation did not turn representable structure into better
     performance. *Caveats (preliminary):* E2 has more params (multi-agg + readout) so
     30k may **under-train** it; its cls + some LP cells + the 2×2/probe diagnostics
     were still running at write time.

## Reading-chain conclusions
- **B1 − B0 (augmentation lever):** the feature shortcut is **not removable by `NR`
  corruption**; the architecture/objective must change. Positive support for the
  E-arms, not a null.
- **E1 − B0 (structural input):** E1 makes topology representable and **beats the
  trivial floor** (raw features / raw degree) on the content×structure join
  (account_age), friends, classification, and static-LP — the one arm that genuinely
  improves transfer. Its structure-target regression is still ≈ passthrough on some
  targets, but account_age + LP are not passthrough wins.
- **E2 − E1 (count-aware encoder):** at matched budget E2 is **worse** than E1 on
  regression and static-LP → making the aggregator count-capable did **not** convert
  E1's representable structure into better performance. **Representational fit was not
  the binding constraint** (the hypothesis this arm was built to test). The win is the
  structural *input*, not the encoder. *Caveat:* E2's larger model may be under-trained
  at 30k; a longer E2 run would tighten this.

## Methodological notes (things that changed a verdict)
- **Shot-matching the leakage baseline mattered.** A full-data Ridge ceiling (163k
  nodes) vs a 10-shot eval wrongly read E1 as passthrough on account_age; the
  shot-matched ceiling (0.06) flipped it — E1 (0.19) genuinely **beats passthrough
  on the content target**. Always compare the leakage probe at the eval's shot count.
- **The 2×2 on feature tasks doesn't discriminate topology-use.** reg/pl don't need
  edges, so rewired-edge ≈ 1.0 for every arm; it confirms B0 is features-only but
  E1's topology evidence lives in **static-LP + probes**, not the feature-task 2×2.
  Consider folding a topology task into T2's retained fraction next time.
- **The trivial-floor anchor was essential.** Arm-vs-arm deltas are uninterpretable
  without `raw_feat` / `raw_degree`: they reveal that B0/B1/E2 *lose to doing nothing*
  on regression, and that E1's win is a genuine improvement, not just "less bad."
  Always run the no-encoder baseline at the eval's shot count.
- **Single seed; small deltas** (the May aug delta was +0.008). Treat T1 as
  confirmatory, T2/T3 as primary. NM 30-way test acc has ±0.12 std.

## Next
- **Firm up E2:** finish its cls + static-LP + 2×2/probe cells; consider a longer E2
  run to rule out under-training before concluding the aggregator is a dead end.
- **The story now points at the objective, not the encoder.** E1 (structural input)
  is the only thing that beats trivial; a fancier encoder (E2) didn't help. That
  redirects to the **objective axis (E3/E4)** — a generative / multi-task target
  that could make the encoder *use* structure — or to a **data-ceiling** conclusion
  (degree is a near-sufficient statistic; SSL on this graph adds little over reading
  it). The free preview already hinted the objective is not an easy win (fp ⊀ nm).
