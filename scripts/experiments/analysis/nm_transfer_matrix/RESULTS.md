# Results — NM transfer matrix (fair single-vs-merged)

**Run date:** 2026-06-28/30 · **Seed:** 0 · **1 seed only.**
Checkpoints: single-source step 50000; merged at **50000 (`@match` = matched total
compute)** and **110000 (`@full` = per-domain exposure)**. Eval: 3-shot, 30-way.
(See ../_cross/NM_MERGED_VS_SINGLE_SUMMARY.md for the cross-experiment view.)

> ⚠️ **Always eval NM at shots ≥ 3.** Zero-shot NM has no support prototypes, so
> accuracy collapses to chance and `roc_auc ≈ 0.5` — this made an earlier eval look
> like every model was random. It was an eval artifact, not a training failure.

## Accuracy (3-shot, 30-way)

```
train             test:ukr   test:covid
ukr (in-domain)   0.5151     0.6142
covid (in-domain) 0.4589     0.6641
merged @match     0.4790     0.6374
merged @full      0.4955     0.6574
```

## AUC (3-shot, 30-way) — near ceiling, less discriminative

```
train             test:ukr   test:covid
ukr               0.9497     0.9741
covid             0.9245     0.9815
merged @match     0.9373     0.9778
merged @full      0.9433     0.9807
```

f1 ≈ accuracy to ~1e-3 (balanced episodes); use `build_auc_matrix.py --metric all`.

## Conclusion

**The original "single-source beats merged cross-domain" inversion does NOT
reproduce** under a fair comparison (identical plain architecture, no augmentation,
correct 3-shot eval). **Merged ≥ single-source cross-domain even at matched compute**
(`@match`):

- test covid: merged @match 0.637 vs single-ukr 0.614 acc (**+0.023**); @full 0.657
- test ukr:   merged @match 0.479 vs single-covid 0.459 acc (**+0.020**); @full 0.496

So the merged advantage is not an artifact of 2× training. The original effect was
almost certainly an artifact of (a) comparing against an *augmented, larger-arch*
merged model rather than a matched one, and/or (b) the degenerate zero-shot eval.

## Caveats

- **1 seed.** Deltas at 3-way are sub-1% (within noise); 30-way deltas are larger
  and consistent but still single-seed. Multi-seed needed for significance.
- **Checkpoint cadence is 0-indexed**, so the final saved checkpoints are 50k (single)
  and 110k (merged), not the nominal 60k/120k. `@match` matches on the actual
  single-source final step (50k), so both sides have identical step counts.
- **Merged-as-test (3rd column)** is supported (loader registered) but slow (33M-node
  graph), so the @match/@full re-eval was run on ukr+covid only; it's not needed for
  the inversion question (the merged test column ≈ the covid column anyway).

## Reproduce

```bash
# (Tucker, prodigy env, repo at /dataMeR1/phil/gfm/prodigy)
cd scripts/experiments/nm_transfer_matrix
STATE_DIR=/dataMeR1/phil/gfm/prodigy/state ./make_model_list.sh   # verify checkpoints
cat model_list.txt
./eval_nm_matrix_tucker.sh --device 0 --continue-on-error          # 3-shot, 30-way
python3 build_auc_matrix.py --log-root /dataMeR1/phil/gfm/prodigy/log \
  --shots 3 --n-way 30 --out-csv auc_matrix_30way.csv
# 3-way view:
./eval_nm_matrix_tucker.sh --device 0 --nm-n-way 3 --continue-on-error
python3 build_auc_matrix.py --log-root /dataMeR1/phil/gfm/prodigy/log --shots 3 --n-way 3
```
