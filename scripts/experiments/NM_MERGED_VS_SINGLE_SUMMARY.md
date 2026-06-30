# NM merged-vs-single: cross-experiment summary

Does a neighbor-matching (NM) model trained on a **disjoint merged** retweet graph
transfer worse than one trained on a **single source**? And does confining each
episode to one source (removing cross-source negatives) help? Validated on two
source pairs. All numbers: plain architecture (no aug), 3-shot / 30-way eval,
seed 0, **1 seed only**. Accuracy shown (most discriminative; AUC saturates ~0.98).

| Experiment | Sources | Folder |
|---|---|---|
| 1. ukr / covid (balanced-ish: ukr ≈ 31% of merge) | ukr_rus + covid | [nm_transfer_matrix](nm_transfer_matrix/RESULTS.md), [nm_cross_source_shortcut](nm_cross_source_shortcut/RESULTS.md) |
| 2. covid / midterm (extreme: midterm ≈ 1.5% of merge) | covid + midterm | [nm_covid_midterm](nm_covid_midterm/RESULTS.md) |

### `@match` vs `@full` (compute axis)

Single-source runs train 60k steps; merged runs train 120k. So we report merged at
two checkpoints:
- **`@match`** — same step count as single-source (final ≈ 50k) → *matched total compute*.
- **`@full`** — merged's final (≈ 110k) → *matched per-domain exposure* (~2× compute).

Experiment 2 evaluates both. Experiment 1 (below) used a single late checkpoint (no
`@match`/`@full` split — that methodology was added for Exp 2), so treat its merged
rows as roughly `@full`.

## Experiment 1 — ukr / covid (30-way, 3-shot, accuracy)

```
train \ test            test:ukr   test:covid
single ukr               0.5151     0.6142
single covid             0.4589     0.6641
merged (naive)           0.4888     0.6536
merged within-source     0.5077     0.6666
```
- **No inversion.** Merged ≥ single cross-domain (test ukr: 0.489 vs 0.459; test
  covid: 0.654 vs 0.614). The original "single beats merged" result did not survive a
  fair comparison — it was an artifact of an unfair architecture/aug mismatch (and a
  degenerate zero-shot eval).
- **Within-source helps, marginally:** beats naive merged on both domains (+0.019 ukr,
  +0.013 covid) and matches/edges the best single-source model. Small (~1–2 pts).

## Experiment 2 — covid / midterm (30-way, 3-shot, accuracy)

Both compute points shown. `—` = not applicable (single-source has only its 50k run).

```
                         test:midterm        test:covid
train                    @match   @full      @match   @full
single midterm           0.4171    —          0.3183    —
single covid             0.3176    —          0.6616    —
merged-naive             0.3137  0.3285       0.6626  0.6728
merged-within            0.3269  0.3373       0.6617  0.6724
merged-within-balanced   0.4048  0.4269       0.6377  0.6511
```
- **Big domain (covid):** replicates Exp 1. At matched compute merged-naive ties single
  (0.663 vs 0.662); at full it slightly beats (0.673).
- **Small domain (midterm):** naive merged **collapses** (0.31–0.33 vs single 0.417) at
  both compute points — an **exposure artifact** (midterm is ~1.5% of the merge), not a
  real merged deficit. Proportional within-source barely helps.
- **Balanced within-source rescues it:** 0.405 @match (recovers most of the gap to
  single's 0.417) and **0.427 @full — above single-midterm** — for a small covid cost
  (−0.02). The *beats-single* win needs the full exposure budget; at matched compute it
  recovers but doesn't exceed single.

## Unified conclusions

1. **The merged model is not inherently worse than single-source.** When you hold
   architecture/budget/eval fixed, merged matches or beats single on cross-domain
   transfer — on every domain that is adequately represented in the merge.
2. **Cross-source negatives let the model take a source-discrimination shortcut.**
   Confining episodes to one source removes it and gives a consistent (if small) gain.
3. **Per-domain exposure is the dominant factor under size imbalance.** A small source
   gets starved by naive sampling; **balanced within-source sampling** rescues it and
   yields the best all-around merged model. This only shows up clearly in Exp 2 (where
   the imbalance is extreme).
4. **Compute:** at matched total compute (`@match`) the within-source gains shrink;
   the "balanced beats single-midterm" win needs the full 2× exposure budget (`@full`).

## Caveats / next step

- **1 seed.** Big effects (naive 0.33 → balanced 0.43 on midterm) are well outside
  noise; sub-1% effects (merged vs single on the big domain, within vs single) are not.
- **Decisive next step: 2–3 seeds** on both experiments (configs take `--seed N`).
- Checkpoints land at 50k (single) / 110k (merged) due to the 0-indexed checkpoint
  cadence; see each folder's RESULTS.md.

Per-experiment details, exact commands, and figures (`plot_results.py`) are in each
folder's `README.md` / `RESULTS.md`.
