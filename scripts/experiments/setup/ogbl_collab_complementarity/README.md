# AA-DC / MLP validation complementarity

Bounded, post-hoc diagnostic authorized after the Collab baseline runs. No training
and no test scoring. Compare the original validation-selected nonlinear cosine MLP
checkpoints for seeds 0, 1, 2 against the pinned AA-DC implementation on all 60,084
official validation positives and 100,000 shared negatives. Both use only the
training graph as graph input. Validation years may set AA-DC's decay reference,
as in upstream. Official negative self-pairs remain unchanged.

First reproduce AA-DC validation Hits@50 (0.673557, rounded log tolerance 5e-7),
and each MLP's saved best validation metric (tolerance one positive / 60,084).
Verify upstream revision, saved selection-file hash, checkpoint hash and epoch,
and original dataset fingerprint. Fail if these gates do not pass.

Primary outputs: positives AA-DC misses but MLP hits at their respective official
50th-negative thresholds; overlap of the top 50 negative identities; and how many
new negatives cross AA-DC's normalized threshold when adding the feature signal.
The union of separate hit masks is only a complementarity upper bound, not a
combined official Hits@50 score.

Predeclared diagnostic rescue curve (fixed before inspecting paired scores):
`max(AA_DC / AA_DC_negative50, alpha * exp(MLP_logit - MLP_negative50))`,
alpha = 0, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2. Recompute the 50th-negative threshold
for every combined score. Report recovered and lost positives, threshold movement,
and new negatives entering the top 50. This is an exploratory validation curve;
its best setting is not an unbiased estimate and is not authorized for test here.
Both inputs already used this validation panel for calibration/selection.

Run from a dedicated Tucker worktree, with the prodigy environment active:

```bash
python scripts/experiments/setup/ogbl_collab_complementarity/run.py \
  --out /dataMeR1/phil/gfm/ogbl_collab_complementarity/validation_v1 \
  --device cpu --threads 8
```

Use `--dry-run` to inspect paths without creating output or scoring. Default W&B is
offline. Score arrays stay under the runtime directory; aggregate JSON, a replay
receipt and findings belong in analysis/baselines/ogbl_collab_complementarity/.
Stop after all three seeds and this fixed curve complete; no tuning expansion.
