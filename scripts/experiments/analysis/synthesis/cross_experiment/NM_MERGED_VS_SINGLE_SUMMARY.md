# NM merged-vs-single: cross-experiment summary

Does a neighbor-matching (NM) model trained on a **disjoint merged** retweet graph
transfer worse than one trained on a **single source**? And does confining each
episode to one source (removing cross-source negatives) help? Validated on two
source pairs. All numbers: plain architecture (no aug), 3-shot / 30-way eval,
seed 0, **1 seed only**. Accuracy shown (most discriminative; AUC saturates ~0.98).

| Experiment | Sources | Folder |
|---|---|---|
| 1. ukr / covid (balanced-ish: ukr ≈ 31% of merge) | ukr_rus + covid | [nm_transfer_matrix](../../transfer/matrices/prodigy_nm/merged_vs_single/nm_transfer_matrix/RESULTS.md), [nm_cross_source_shortcut](../../transfer/ablations/prodigy_nm/episode_sampling/nm_cross_source_shortcut/RESULTS.md) |
| 2. covid / midterm (extreme: midterm ≈ 1.5% of merge) | covid + midterm | [nm_covid_midterm](../../transfer/matrices/prodigy_nm/merged_vs_single/nm_covid_midterm/RESULTS.md) |

### `@match` vs `@full` (compute axis)

Single-source runs train 60k steps; merged runs train 120k. So we report merged at
two checkpoints:
- **`@match`** — same step count as single-source (final ≈ 50k) → *matched total compute*.
- **`@full`** — merged's final (≈ 110k) → *matched per-domain exposure* (~2× compute).

Both experiments evaluate both compute points.

All tables: rows = train regime (merged at `@match`/`@full`), columns = test domain ×
compute point. `—` = not applicable (single-source has only its 50k run). The **`*`
column is held-out (OOD)** — no model in that experiment trained on it. f1 ≈ accuracy
throughout (balanced n-way episodes); included for completeness.

## Experiment 1 — ukr / covid (30-way, 3-shot) · held-out = midterm

accuracy
```
                         test:ukr        test:covid      test:midterm* (held-out)
train                    @match  @full   @match  @full   @match  @full
single ukr               0.5151   —      0.6142   —      0.3079   —
single covid             0.4589   —      0.6641   —      0.3126   —
merged proportional      0.4790 0.4955   0.6374 0.6574   0.2968 0.3048
merged within-source     0.4998 0.5090   0.6592 0.6698   0.3069 0.3144
```
f1
```
                         test:ukr        test:covid      test:midterm* (held-out)
train                    @match  @full   @match  @full   @match  @full
single ukr               0.5151   —      0.6142   —      0.3081   —
single covid             0.4591   —      0.6641   —      0.3143   —
merged proportional      0.4790 0.4954   0.6374 0.6574   0.2976 0.3052
merged within-source     0.4998 0.5090   0.6592 0.6698   0.3072 0.3146
```
roc_auc
```
                         test:ukr        test:covid      test:midterm* (held-out)
train                    @match  @full   @match  @full   @match  @full
single ukr               0.9497   —      0.9741   —      0.8840   —
single covid             0.9245   —      0.9815   —      0.8837   —
merged proportional      0.9373 0.9433   0.9778 0.9807   0.8746 0.8755
merged within-source     0.9447 0.9472   0.9811 0.9822   0.8835 0.8847
```
- **No inversion — even at matched compute.** Merged ≥ single cross-domain at `@match`
  (test ukr: 0.479 vs single-covid 0.459, +0.020; test covid: 0.637 vs single-ukr
  0.614, +0.023). The original "single beats merged" result was an artifact of an
  unfair architecture/aug mismatch (and a degenerate zero-shot eval).
- **Within-source > proportional at both compute levels** (+0.02 @match, +0.012–0.016
  @full).
- **Held-out (midterm):** all ukr/cov models land ~0.30–0.31 (vs an in-domain
  midterm-trained model's 0.417 — see Exp 2). Merging gives **no OOD bonus**; merged
  within-source @full (0.314) is marginally best but within noise.

## Experiment 2 — covid / midterm (30-way, 3-shot) · held-out = ukr

accuracy
```
                         test:midterm    test:covid      test:ukr* (held-out)
train                    @match  @full   @match  @full   @match  @full
single midterm           0.4171   —      0.3183   —      0.2256   —
single covid             0.3176   —      0.6616   —      0.4625   —
merged-naive             0.3137 0.3285   0.6626 0.6728   0.4598 0.4610
merged-within            0.3269 0.3373   0.6617 0.6724   0.4586 0.4586
merged-within-balanced   0.4048 0.4269   0.6377 0.6511   0.4480 0.4476
```
f1
```
                         test:midterm    test:covid      test:ukr* (held-out)
train                    @match  @full   @match  @full   @match  @full
single midterm           0.4171   —      0.3183   —      0.2256   —
single covid             0.3191   —      0.6616   —      0.4626   —
merged-naive             0.3141 0.3285   0.6626 0.6728   0.4598 0.4611
merged-within            0.3271 0.3374   0.6616 0.6724   0.4588 0.4588
merged-within-balanced   0.4048 0.4269   0.6377 0.6511   0.4480 0.4476
```
roc_auc
```
                         test:midterm    test:covid      test:ukr* (held-out)
train                    @match  @full   @match  @full   @match  @full
single midterm           0.9260   —      0.8792   —      0.7935   —
single covid             0.8860   —      0.9813   —      0.9257   —
merged-naive             0.8849 0.8899   0.9810 0.9827   0.9249 0.9240
merged-within            0.8943 0.8984   0.9814 0.9828   0.9243 0.9244
merged-within-balanced   0.9231 0.9291   0.9780 0.9796   0.9216 0.9212
```
- **Big domain (covid):** replicates Exp 1 — merged ties single @match (0.663 vs 0.662),
  slightly beats @full.
- **Small domain (midterm):** naive merged **collapses** (0.31–0.33 vs single 0.417) —
  an **exposure artifact** (midterm ~1.5% of the merge). **Balanced within-source
  rescues it:** 0.405 @match, **0.427 @full — above single-midterm (0.417)**, for a
  small covid cost (−0.02). The beats-single win needs the full exposure budget.
- **Held-out (ukr):** transfer is carried by covid — single-covid 0.463 ≈ merged-naive
  0.461; single-midterm transfers terribly (0.226). Merging gives **no OOD bonus**, and
  **balanced is the *weakest* OOD (0.448)** because it down-weights the covid signal
  that carries cross-graph transfer. (In-domain ukr ceiling: 0.515.)

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
5. **No generalization bonus from merging (held-out / OOD).** On a graph *no* model
   trained on, merged transfers no better than the best single source — transfer is
   carried by the largest/most-similar source (covid drives ukr; combining adds
   nothing). And **balanced sampling, best in-distribution, is slightly worst OOD**,
   because rebalancing down-weights the source that carries cross-graph signal. So the
   in-distribution wins (no inversion; balanced rescues small domains) do **not** imply
   better transfer to unseen graphs.

## Caveats / next step

- **1 seed.** Big effects (naive 0.33 → balanced 0.43 on midterm) are well outside
  noise; sub-1% effects (merged vs single on the big domain, within vs single) are not.
- **Decisive next step: 2–3 seeds** on both experiments (configs take `--seed N`).
- Checkpoints land at 50k (single) / 110k (merged) due to the 0-indexed checkpoint
  cadence; see each folder's RESULTS.md.

Per-experiment details, exact commands, and figures (`plot_results.py`) are in each
folder's `README.md` / `RESULTS.md`.
