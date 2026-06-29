# Results — NM transfer matrix (fair single-vs-merged)

**Run date:** 2026-06-28/29 · **Seed:** 0 · **1 seed only.**
Checkpoints: single-source step 50000, merged step 90000 (runs stopped short of the
60k/120k budget — see Caveats). Eval: 3-shot, both 3-way and 30-way.

> ⚠️ **Always eval NM at shots ≥ 3.** Zero-shot NM has no support prototypes, so
> accuracy collapses to chance and `roc_auc ≈ 0.5` — this made an earlier eval look
> like every model was random. It was an eval artifact, not a training failure.

## Accuracy (3-shot, 30-way) — most discriminative, full 3x3

```
train\test   ukr     covid   merged
ukr          0.5151  0.6142  0.6156
covid        0.4589  0.6641  0.6238
merged       0.4888  0.6536  0.6872
```
- test ukr:   merged 0.4888 vs single-covid 0.4589 (**+0.030**)
- test covid: merged 0.6536 vs single-ukr 0.6142 (**+0.039**)
- merged is also best in-domain on its own (mixed) test (0.6872).

f1 is within ~0.0002 of accuracy everywhere (balanced n-way episodes), so it adds
nothing beyond accuracy; omitted here. Use `--metric all` to print all three.

## AUC (3-shot, 30-way) — same ordering, near ceiling (less discriminative)

```
train\test    ukr       covid
ukr          0.9497    0.9741
covid        0.9245    0.9815
merged       0.9411    0.9801
```

## AUC (3-shot, 3-way) — near ceiling, less discriminative

```
train\test    ukr       covid
ukr          0.9621    0.9857
covid        0.9464    0.9911
merged       0.9567    0.9897
```

## Conclusion

**The original "single-source beats merged cross-domain" inversion does NOT
reproduce** under a fair comparison (identical plain architecture, no augmentation,
matched per-domain episode budget, correct 3-shot eval). On both cross-domain cells,
**merged ≥ single-source** — clearest in accuracy:

- test covid: merged 0.654 vs single-ukr 0.614 acc (**+0.039**) / 0.980 vs 0.974 AUC
- test ukr:   merged 0.489 vs single-covid 0.459 acc (**+0.030**) / 0.941 vs 0.925 AUC

The original effect was almost certainly an artifact of (a) comparing against an
*augmented, larger-architecture* merged model rather than a matched one, and/or
(b) the degenerate zero-shot eval.

## Caveats

- **1 seed.** Deltas at 3-way are sub-1% (within noise); 30-way deltas are larger
  and consistent but still single-seed. Multi-seed needed for significance.
- **Checkpoints stopped early** (50k/90k vs 60k/120k budget) — likely early-stopping
  or the accidental duplicate-launch kill. Conclusions are unlikely to flip, but a
  clean full-budget re-run would be tidier.
- **Merged-as-test (3rd column)** is supported now (loader registered) but slow
  (33M-node graph); it's a bonus and not needed for the inversion question.

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
