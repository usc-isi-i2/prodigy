# Partial Cross-Source Episodes — Results

**Verdict: remedy #4 does NOT help. Pure within-source (p=0) is best-or-tied on every
domain and metric; adding cross-source episodes monotonically hurts single-source
transfer. No interior optimum.** (1 seed; see caveats.)

Testbed: ukr+covid disjoint merge, plain arch / no aug, `n_way=30 n_shots=3 n_query=4
n_hop=1`, 120k episodes, seed 0, confined-source weighting `proportional`. Only
`neighbor_sampling_cross_source_prob = p` varies. Eval: frozen encoder, NM 3-shot /
30-way on the held-out single-source graphs (ukr_rus_twitter, covid19_twitter), final
110k checkpoint. Full numbers in `sweep.csv`; raw dump in `RESULTS.txt`.

## Accuracy (discriminative metric; f1 ≈ accuracy)

| p | test:ukr | test:covid |
|---|---|---|
| **0.00 (within)** | 0.5120 | **0.6710** |
| 0.10 | **0.5130** | 0.6707 |
| 0.25 | 0.5054 | 0.6654 |
| 0.50 | 0.4983 | 0.6608 |
| 1.00 (naive) | 0.4984 | 0.6607 |

## ROC-AUC (near-ceiling, non-discriminative — reported for completeness)

| p | test:ukr | test:covid |
|---|---|---|
| 0.00 (within) | 0.9488 | 0.9827 |
| 0.10 | 0.9486 | 0.9825 |
| 0.25 | 0.9463 | 0.9820 |
| 0.50 | 0.9436 | 0.9813 |
| 1.00 (naive) | 0.9437 | 0.9811 |

## Reading

1. **No real interior optimum.** `build_sweep.py`'s mechanical argmax flags ukr p=0.10 as
   "interior best," but it beats within-source by **0.0010** (0.5130 vs 0.5120) — noise at
   1 seed with a ±0.09 per-episode std — and covid points the other way (within-source
   best). The interior points do not beat pure within-source. (The verdict logic was
   tightened after this run to require a margin + cross-domain agreement before claiming an
   interior win; it now reports "no interior optimum" here.)
2. **Robust signal (agrees across 2 domains × 3 metrics, monotone):** transfer degrades as
   `p` rises. within (p=0) is best-or-tied everywhere; even 25% mixing measurably hurts
   (ukr 0.505 vs 0.512; covid 0.665 vs 0.671); the damage saturates by ~50% (p=0.5 ≈ p=1).
3. **within (p=0) beats naive (p=1)** by +0.014 ukr / +0.010 covid acc (AUC also higher on
   both) — **replicates** the prior cross-source-shortcut finding (`../../nm_cross_source_shortcut`,
   which saw +0.019 / +0.013). The p=0 endpoint reproduced the known within-source numbers
   (ukr 0.512 acc / 0.949 AUC), confirming checkpoint loading + eval features are correct.

**Interpretation.** The cross-source shortcut is real, but you can't beat simply removing
all of it. There is no "the model needs *some* source discrimination" sweet spot at this
budget/pair — mixing only reintroduces the shortcut and costs transfer. **Recommendation:
keep pure within-source confinement.**

## Caveats

- **1 seed.** Full spread is ~0.014 acc (ukr) / ~0.010 (covid) — small. What makes the
  *direction* trustworthy is the monotone agreement across both domains and all three
  metrics; the exact p=0-vs-0.10 ordering is a coin-flip and should not be read as a win.
- Eval episodes are seeded per-split (see the eval-seed note in the study), so all arms saw
  identical eval episodes — the cross-arm comparison is apples-to-apples. Cross-*domain*
  agreement (ukr and covid pointing the same way) is the reliability signal used here.
- A 3–5 seed sweep would be needed to put error bars on the (small) within-vs-naive gap,
  but is unlikely to resurrect an interior optimum given how flat/monotone the curve is.

## Reproduce

```bash
# train (5 arms) then, from the worktree:
scripts/experiments/setup/sampling_improvements/partial_cross_source/make_model_list.sh
scripts/experiments/setup/sampling_improvements/partial_cross_source/eval_tucker.sh --device 0
python3 scripts/experiments/setup/sampling_improvements/partial_cross_source/build_sweep.py \
  --log-root <repo>/log --shots 3 --n-way 30 --metric all --out-csv .../sweep.csv
```
Note: `eval_tucker.sh` now defaults `--data-root /dataMeR1/phil/data` (the single-source
GTE graphs live on `/dataMeR1`, not the runner's `/dataMeR2` default — the bug that made
the first automated eval produce zero results).
