# MT transfer pilot analysis

Matched 5×5 source-to-target classification transfer matrices for supervised MT and
equal-compute NM+MT at 900 optimizer updates. Evaluation is 3-shot, seed 0, with the
catalog's native class count (binary for four graphs and 30-way for Facebook).

Run on Tucker after evaluation:

```bash
MPLBACKEND=Agg python analyze.py --log-root /dataMeR1/phil/gfm/prodigy-mtfast/log
```

The older NM matrix is intentionally not subtracted here because it used a different
10-shot evaluation protocol. A matched NM-only arm is required for a defensible
three-way causal comparison.
