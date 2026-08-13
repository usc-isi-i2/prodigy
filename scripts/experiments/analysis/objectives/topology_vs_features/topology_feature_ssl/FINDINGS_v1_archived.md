# topology_feature_ssl — Findings (B0 / B1 / E1 / E2 / E2b vs trivial floor)

**Date:** 2026-07-09; **matched-40k eval + E2b drop-BN retry completed 2026-07-11.**
**Scope:** the 5 encoder/augmentation arms (B0/B1/E1/E2 + the E2b drop-BN retry) at a
matched **40k** budget, full diagnostics + trivial-floor baselines. E3–E4 (objective
axis) deferred. See `README.md` (plan), `RESULTS_matched40k.md` (the final rendered 40k
tables — supersede the 30k tables here), `EXECUTION.md` (ops).

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

**Budget — matched 40k (done).** All arms are compared at a true `state_dict_40000.ckpt`
(the NM regression peak; NM anti-scales past ~40–60k). E2/E2b needed `epochs:5`, not 4,
because of a trainer off-by-one: `trange(steps=epochs*10000)` runs `e=0…steps-1` and
checkpoints at `e%10000==0`, so `epochs:4` (steps 40000) saves its last ckpt at **30000**
(same reason B0/E1 top out at 110k, not 120k). The final numbers are in
[`RESULTS_matched40k.md`](RESULTS_matched40k.md) and summarized in **Matched-40k results**
below; the older per-section 30k tables further down are superseded and kept only for the
reading-chain narrative.

**Trivial floor (the absolute anchor):** `raw_feat` = 10-shot linear probe on the
raw 768-d bio embedding (no encoder); `raw_degree` = 10-shot probe on the raw
`directed3` features (leakage). An arm only *"improves performance"* if it beats
these — not just the other arms.

## Headline
**No single encoder-axis arm transfers to BOTH task families — they split.** E1
(structural **INPUT**) wins the **feature** tasks: regression positive on all six targets
(followers 0.19, account_age 0.12 — clears the trivial floor) plus the count/in-degree
probes. E2 (count-aware **ENCODER**) wins the **topological** task: static-LP **0.76**, the
best of any arm — the multi-aggregation encoder genuinely helps link prediction — but its
regression is at/below zero. E2b (drop the conv BatchNorm) **confirms the named "BN washes
out the sum's count magnitude" mechanism** at the probe level (count 0.59→0.66, out-deg
0.58→0.71) **yet crashes static-LP 0.76→0.40** and leaves regression ~0 — a clean
"representable ≠ used" dissociation, and *not* a fix. **Neither E2 nor E2b clears the JOINT
(feature AND topology) bar; E1 and E2 are complementary** — exactly the case for changing
the **objective** (E4), not the encoder. B1 (`NR` augmentation) backfires; B0/NM is
features-only.

## Matched-40k results (final — supersede the 30k tables below)

Means over the focused datasets, all arms at a true 40k ckpt (full tables:
[`RESULTS_matched40k.md`](RESULTS_matched40k.md)). Trivial floor: `raw_feat` (bio, no
encoder), `raw_degree` (leakage on the directed3 inputs).

| arm | reg followers / age (Spearman) | static-LP (ROC-AUC) | cls (twibot20) | probes count / out-deg |
|---|---|---|---|---|
| raw_feat | 0.20 / 0.02 | — | 0.56 | — |
| raw_degree | 0.16 / 0.01 | — | — | — |
| B0 | 0.03 / −0.06 | 0.68 | 0.61 | 0.48 / 0.52 |
| B1 | −0.13 / −0.13 | 0.34 | 0.61 | 0.53 / 0.52 |
| **E1** | **0.19 / 0.12** | 0.66 | 0.60 | **0.67** / 0.52 |
| **E2** | −0.07 / −0.07 | **0.76** | 0.59 | 0.59 / 0.58 |
| E2b | −0.04 / 0.05 | 0.40 | 0.60 | 0.66 / **0.71** |

Reads: **E1** is the only arm that beats the trivial floor on regression (feature
transfer). **E2** is the only arm that lifts static-LP above the baselines (topological
transfer) — but not features. **E2b**'s BN-drop raises the count/degree probes but tanks
LP: the count magnitude became linearly *decodable* without becoming *usable*.
Classification is flat across all arms (~0.60 on twibot20, ~0.97 on election2020) and does
not discriminate. The T2 2×2 is uninformative here (near-zero feature-task denominators
make the retained fraction explode) — topology evidence lives in static-LP + probes.

## Findings (original 30k-preliminary notes — see Matched-40k results above for the final read)

> ⚠️ **Two claims below flipped at the true 40k budget** and are corrected above: (a) E2's
> **static-LP is 0.76 at 40k** (the best arm), not the ~0.4 seen at 30k — the count-aware
> encoder *does* help the topological task; (b) Finding 5's "why sum didn't buy counting"
> (BatchNorm washout) is now **tested by E2b** — dropping the conv BN *does* lift the count/
> degree probes (count 0.59→0.66, out-deg 0.58→0.71) but crashes static-LP 0.76→0.40, so it
> is a real mechanism but not a usable fix. Everything else below still holds.

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
     budget. And the **capability probes explain why** — it's *not* "representable but
     unused," the aggregator **still doesn't make structure representable**:

     | probe AUC | count | in-deg | out-deg | existence | conj |
     |---|---|---|---|---|---|
     | E2@10k | 0.56 | 0.52 | 0.61 | 0.59 | 0.60 |
     | E2@30k | 0.59 | 0.53 | 0.60 | 0.62 | 0.61 |
     | E1@110k | 0.64 | 0.59 | 0.53 | 0.56 | 0.52 |

     E2's probes barely move 10k→30k (converged, not under-trained) and sit **≤ E1 on
     count/in-degree** — the exact primitives sum-aggregation was meant to lift. E2
     only edges E1 on out-deg/existence/**conjunction** (presence/binding, which max
     helps). Structure caps at ~0.6 for *both* mean and multi-agg encoders.
   - **Why sum didn't buy counting:** `sum` ∝ degree, but the conv **BatchNorms** its
     output and projects the 3× aggregate down — BN rescales to unit variance, washing
     out the very magnitude that encodes count. The count signal is created by
     sum-aggregation and then normalized away. Concrete + fixable (true PNA uses
     degree-scalers to avoid this; or drop BN on the aggregate). *Preliminary:* E2's
     cls + some LP + the full 2×2/probe run were still in progress at write time.

## Reading-chain conclusions
- **B1 − B0 (augmentation lever):** the feature shortcut is **not removable by `NR`
  corruption**; the architecture/objective must change. Positive support for the
  E-arms, not a null.
- **E1 − B0 (structural input):** E1 makes topology representable and **beats the
  trivial floor** (raw features / raw degree) on the content×structure join
  (account_age), friends, classification, and static-LP — the one arm that genuinely
  improves transfer. Its structure-target regression is still ≈ passthrough on some
  targets, but account_age + LP are not passthrough wins.
- **E2 − E1 (count-aware encoder):** the two arms **split by task family** at matched 40k.
  E2 **wins static-LP (0.76 vs E1's 0.66)** — the multi-aggregation encoder genuinely helps
  the topological task — but its **regression is at/below zero** (E1 wins the feature tasks).
  The encoder buys topology, the input buys features; **neither buys both**. Probes stay
  ~0.6 (E2 edges E1 on out-deg/existence/conjunction; E1 leads count/in-degree via its
  injected directed-degree inputs). So "representational fit" partly *is* supported — but
  only for the topological task, and it does not generalize to features.
- **E2b − E2 (drop the conv BatchNorm):** confirms the named mechanism but is **not a fix**.
  Removing the BN that rescales the sum's magnitude lifts the count/degree probes (count
  0.59→0.66, in-deg 0.51→0.56, out-deg 0.58→0.71) — so BN *was* washing out the count
  magnitude at the linear-probe level — yet **static-LP crashes 0.76→0.40** and regression
  stays ~0. Making counts *decodable* did not make them *usable*; BN was doing real work for
  the task-useful representation. ⇒ **the encoder axis is exhausted** — no encoder variant
  clears the joint (feature + topology) bar.

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

## Next (matched-40k + E2b done 2026-07-11 — single-seed throughout)

**The encoder axis is exhausted; the fork is the objective (E4).** E1 (input) wins the
feature tasks, E2 (encoder) wins the topological task, and **no encoder variant wins both**;
E2b's BN-drop confirms the count mechanism at the probe level but trades static-LP for it.
Making structure *representable* did not make NM *use* it jointly — that is an **objective**
problem, not a capacity one. The design's expected prior ("objective is the binding
constraint") is what the data now points to.

1. **Build E4 — multi-task MFR ⊕ directed-LP ⊕ structural-property prediction** on **E2's
   encoder** (it already wins static-LP), matched 40k (`epochs:5`; clone `configs/E2.yaml`).
   The LP head must read the **original directed edges / `edge_attr` sign**, not the
   symmetrized sampling adjacency (else "directed LP" isn't directed). Hypothesis: an
   explicit topological + generative objective unifies E1's feature transfer with E2's
   topological transfer in one representation — the joint bar no encoder-axis arm cleared.
   This is the arm that must clear it: feature tasks up **and** static-LP up together.
2. **E3 (MFR alone) is low-priority** — the free preview already showed fp ⊀ nm (−0.016 on
   regression). Keep it as an **E4 ablation**, not a standalone arm.
3. **Deferred encoder coda (A-arm, not the main line):** a degree-scaler PNA (reintroduce
   degree magnitude *without* BN's global rescale) might recover E2b's probe gain without
   the LP collapse — but E2 already wins LP and the bottleneck is *joint* transfer, so this
   is optional.
4. **Data-ceiling caveat:** if E4 also fails the joint bar, the bounded-negative conclusion
   is that degree is a near-sufficient statistic and social-graph SSL at this scale/depth is
   feature-limited (→ deferred levers: depth A3, positional encodings, temporal A5).
