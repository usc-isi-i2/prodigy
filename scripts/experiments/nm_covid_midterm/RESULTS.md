# Results — NM covid/midterm validation

**Run date:** 2026-06-29/30 · **Seed:** 0 · **1 seed only.**
Checkpoints: single-source final = step 50000 (60k budget; last checkpoint lands at
50k due to the 0-indexed cadence); merged final = step 110000 (120k budget). Eval:
3-shot, 30-way, on covid + midterm + merged.

Each **merged** model is evaluated at two checkpoints:
- **`@match`** = step 50000 → same step count as the single-source runs (**matched
  total compute** — the apples-to-apples comparison).
- **`@full`** = step 110000 → merged's final (**matched per-domain exposure**, ~2x compute).

> midterm is tiny: ~0.34M nodes vs covid's ~23M → **midterm is ~1.5% of the merge**.
> So naive/proportional sampling barely trains on midterm; "balanced" within-source
> gives each source equal episode share.

## Accuracy (most discriminative; f1 ≈ accuracy to ~1e-3)

```
train \ test              midterm   covid     merged
single midterm             0.4171    0.3183    0.3231
single covid               0.3176    0.6616    0.6653
merged-naive   @match      0.3137    0.6626    0.6669
merged-naive   @full       0.3285    0.6728    0.6764
merged-within  @match      0.3269    0.6617    0.6654
merged-within  @full       0.3373    0.6724    0.6766
merged-within-bal @match   0.4048    0.6377    0.6429
merged-within-bal @full    0.4269    0.6511    0.6554
```

## AUC (near ceiling on covid; less discriminative)

```
train \ test              midterm   covid     merged
single midterm             0.9260    0.8792    0.8806
single covid               0.8860    0.9813    0.9816
merged-naive   @match      0.8849    0.9810    0.9815
merged-naive   @full       0.8899    0.9827    0.9832
merged-within  @match      0.8943    0.9814    0.9817
merged-within  @full       0.8984    0.9828    0.9831
merged-within-bal @match   0.9231    0.9780    0.9783
merged-within-bal @full    0.9291    0.9796    0.9801
```

(The `merged` test column ≈ the covid column — the merged test set is 98.5% covid —
so it is not independently informative.)

## Conclusions

The story splits by domain size, and the answer depends on the compute budget.

**Big domain (covid):** the ukr/cov result replicates.
- Matched compute (`@match`): merged-naive (0.663) ≈ single-covid (0.662) — tied, no
  inversion.
- Full exposure (`@full`): merged-naive (0.673) and within (0.672) slightly *beat*
  single-covid (0.662).

**Small domain (midterm):** exposure dominates.
- Naive merged **collapses** to 0.31–0.33 (≈ the covid-only model's transfer), far
  below single-midterm (0.417) — the "merged is worse" pattern, here clearly an
  **exposure artifact** (midterm seen ~1.5% of episodes). Proportional within-source
  barely helps (still ~1.5% midterm episodes).
- **Balanced within-source rescues it:** 0.405 at matched compute, **0.427 at full —
  beating single-midterm (0.417)**. Equal episode share lets the small domain recover.

**Balanced is the best all-around merged model:** big win on the starved small domain
(+0.10 acc on midterm vs naive) for a small big-domain cost (−0.02 acc on covid). The
cost on covid comes from giving covid 50% of episodes instead of 98.5%.

**Matched-compute vs full-exposure:** `@full` beats `@match` by ~0.01 everywhere (more
training helps both). The headline "balanced merged beats single-midterm" is a
**`@full` (2× compute)** result; at **matched compute**, balanced *recovers* most of
the naive→single gap (0.31→0.40 vs 0.42) but does not exceed single.

## Caveats

- **1 seed.** The big effect (naive 0.33 → balanced 0.43 on midterm) is well outside
  noise; the small ones (balanced vs single-midterm ±0.01, merged vs single on covid
  ±0.01) are not — need multiple seeds to call those.
- AUC saturates on covid (~0.98); read accuracy.
- Checkpoints are 50k/110k, not the nominal 60k/120k (0-indexed checkpoint cadence;
  see the note above). `@match` matches on the *actual* single-source final step, so
  both sides have identical step counts.

## Reproduce

```bash
# (Tucker, prodigy env, repo at /dataMeR1/phil/gfm/prodigy)
cd scripts/experiments/nm_covid_midterm
STATE_DIR=/dataMeR1/phil/gfm/prodigy/state ./make_model_list.sh   # 8 entries: single + merged @match/@full
cat model_list.txt
./eval_tucker.sh --device 0 --continue-on-error                   # 3-shot, 30-way; 8 models x 3 test sets
python3 build_matrix.py --log-root /dataMeR1/phil/gfm/prodigy/log \
  --shots 3 --n-way 30 --metric all --out-csv matrix.csv
```
