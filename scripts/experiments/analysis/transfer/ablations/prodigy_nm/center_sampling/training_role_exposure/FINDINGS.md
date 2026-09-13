# Training-role exposure under the sampler repair

## Evidence contract

This audit compares the old and repaired Facebook sampler at fixed episode budgets.
The four committed `training_user_role_counts.csv` files record per-user anchor,
support, and query selections at 100 and 2,500 training steps. Eight metric JSON
files preserve the available old/fixed model-by-evaluator checks. The producing
Tucker worktrees were `prodigy-roleexposure` at `394475e337` and
`prodigy-roleexposure-old` at `ee8ab031f6`.

## Finding

At 2,500 steps, both samplers made 75,000 anchor, 225,000 support, and 300,000
query selections. The repair broadened support exposure from 14,498 to 43,503
unique users and query exposure from 21,946 to 46,468 unique users. Total unique
users increased from 33,217 to 52,066. Maximum support frequency fell from 4,190
to 1,493 and maximum query frequency from 3,108 to 1,900. The 100-step audit points
in the same direction: total unique users increased from 13,313 to 14,072 and the
maximum combined role count fell from 288 to 144.

The sampler repair therefore materially broadened and flattened training-role
exposure at a fixed episode budget.

## Performance limitation

The crossed performance checks are exploratory and protocol-incomparable for a
causal performance claim. Test accuracy/AUC were 0.73250/0.98869 for old model and
old evaluator, 0.69550/0.98387 for old model and fixed evaluator,
0.68375/0.98250 for fixed model and old evaluator, and 0.69092/0.98424 for fixed
model and fixed evaluator. Model training and evaluation-panel changes are not held
fixed in one admissible contrast, so these values do not identify a downstream
effect of the sampler repair.

The files under `data/` are the canonical preserved evidence. Checkpoints, W&B
runs, score dumps, and console logs were intentionally excluded.
