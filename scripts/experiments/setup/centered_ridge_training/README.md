# Strengthened direct-readout training comparator

## Question fixed before outcomes

The earlier isolation pilot found that allowing native-inference gradients to
reach the encoder improved a subsequently fitted intermediate ridge readout.
However, its direct ridge-training comparator used uncentered features and a
fixed logit scale of one. This experiment tests that alternative explanation,
not a new architecture or a claimed need for learned inference.

## One objective, two existing schedules

- Start from the isolation code baseline `d5843502` in a separate worktree.
- Use the same rung-four sources, blocked/interleaved schedules, training seed0,
  source-private episode seeds380100/480100, encoder, optimizer, learning rate,
  regularization, batch size, and2,500-update budget as the eight completed arms.
- At U1, independently within each task, subtract the support mean from both
  support and query features; then row-L2 normalize. Never use the query mean.
- Fit differentiable dual ridge with lambda1, one-hot support targets and no
  intercept. Optimize query cross-entropy on pretraining episodes.
- Multiply ridge logits by a learned positive scale, using the model's existing
  `exp(logit_scale)`, initialized identically to native training at1/0.07.
  Reuse that parameter: no extra initialization draws or model-state entries.
- No native loss reaches the encoder or contributes to the objective. Retain
  the ordinary detached native forward only for implementation compatibility.
- Do not tune temperatures, loss weights, steps or source mixtures on targets.

The new temperature is learned on pretraining data, not calibrated on target
queries. At evaluation, apply the same fixed support-centered ridge rule as the
existing deployment comparison; do not use the learned training temperature to
select a favorable target NLL. Report its trained value as a diagnostic.

## Required checks and comparison

Run both20-update smoke arms before the full run. Verify exact initial model
states against the previous isolation run and actual consumed-episode payloads
against the matching schedules. Do not require bit-exact CUDA optimization
trajectories: the earlier identical-objective null measured small reduction
differences amplified by AdamW. Unit tests must establish support-only centering,
gradient flow to the encoder and logit scale, and absence of a native-loss path.

Reuse the existing original/fresh target caches and fixed centered deployment
readout. Report all five targets; retain the previously defined supported-four
panel (COVID Political, Election2020, Facebook page-reference, TwiBot20).
Compare primarily with native-trained and joint-trained encoders using that same
readout, with the previous isolated/ridge-only arms as explanatory references.
Report accuracy, macro-F1, AUC and NLL, and both schedules separately.

## Decision

If strengthened direct training closes the prior gap, abandon the claim that
the old pilot establishes a special representation benefit from learned native
inference. If a substantial gap remains, retain only the bounded conclusion
that this strengthened direct objective does not reproduce the tested contextual
training benefit. Neither outcome establishes universal necessity, a novel
auxiliary-head principle, or an8+ contribution by itself.

From the dedicated Tucker worktree, with `prodigy` activated:

```bash
python -m scripts.experiments.setup.centered_ridge_training.run \
  --gpu 3 --steps 20 \
  --output /dataMeR1/phil/gfm/prodigy-centered-ridge-training/log/centered_ridge_training_smoke
```

Dry-run is the default; add `--execute` after checking GPU availability. The full
plan uses `--steps 2500` and a distinct output directory. Do not reuse an existing
output or pull this worktree while either arm is running.

After both smoke arms complete:

```bash
python -m scripts.experiments.setup.centered_ridge_training.verify \
  --root /dataMeR1/phil/gfm/prodigy-centered-ridge-training/log/centered_ridge_training_smoke \
  --reference /dataMeR1/phil/gfm/prodigy-encoder-solver-isolation/log/isolation_full_20260907
```

Repeat the audit with the full output directory after training. The smoke audit
matches each source's consumed prefix, not the first20 global steps of a longer
blocked schedule. It checks exact initial model tensors and a finite, positive,
updated logit scale. Payload matching covers IDs, roles and sampled topology;
do not describe it as a bytewise check of every input feature.
