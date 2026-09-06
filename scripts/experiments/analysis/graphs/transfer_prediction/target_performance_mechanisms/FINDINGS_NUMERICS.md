# Same seed and inputs do not guarantee the same trained model

6 September 2026. **Complete full-run reproducibility stress test, not a transfer method.**

Two default CPU executions with the same Hong Kong seed-2 initialization and
all **2,500 complete training inputs** end .0719/.0629 AUC apart on COVID
Political's original/fresh streams. Deterministic executions produce identical
model/optimizer/RNG/sampler states at every saved checkpoint and identical
predictions across all five targets, both streams, and all 17 decoders.

This demonstrates material experimental variability beyond training data,
initialization seed, and sampled examples. It does not explain the general
Ukraine/COVID donor advantage, estimate typical run variability, or make
deterministic execution a performance-improvement method.

## Design and validity

The case was selected before these outcomes because Hong Kong seed 2 had the
largest earlier same-seed discrepancy. Four models: two default executions and
two with deterministic algorithms enabled, no decoder source edits. All use
the earlier free-readout control's resolved recipe, two loader workers, eight
CPU threads per trainer, and the same audit hook. GPUs are hidden. Initial
model, optimizer, and parent random states are exact; every complete consumed
input matches the recorded earlier free-control input at the same update:
**10,000 successful input comparisons** across the four runs.

All four terminal checkpoints are evaluated; there is no target-selected step
or best-repeat selection. Saved states are 0/100/300/900/2500. Original/fresh
test streams reproduce the established cached inputs, with no raw-input probe
differences. All 680 diagnostic cells and 40 full-model cells are present.
The local validator independently checks state/hash agreement, input and model
provenance, decoder grids, and all 20 saved-logit comparisons.

The declared prediction—exact deterministic saved states and target logits—
passes. Default model/optimizer states differ at saved step 100 and every
later saved step; their parent RNG and sampler states remain exact.
Deterministic states match at all five saved steps, and all 5,440 paired
batch/decoder prediction tensors match exactly across targets and streams.

## All final full-model AUCs

Each entry is original / fresh. The deterministic repeats are identical.

| Target | Default repeat 0 | Default repeat 1 | Deterministic repeats 0 and 1 |
|---|---|---|---|
| COVID Political | .8337 / .8603 | .7618 / .7974 | .8118 / .8439 |
| Election | .9811 / .9850 | .9705 / .9636 | .9767 / .9869 |
| Facebook Page Reference | .6714 / .6691 | .7100 / .6897 | .6870 / .7101 |
| TwiBot20 | .6574 / .6534 | .6656 / .6670 | .6384 / .6271 |
| Ukraine Suspended | .4810 / .5369 | .4795 / .5114 | .5021 / .5330 |

Default-repeat differences have different signs across tasks: one execution
is not uniformly better. Deterministic training does not uniformly improve AUC
either. These two repeats are not an uncertainty interval, and the two
evaluation streams are not independent domains or training seeds.

## What this establishes, and what it does not

Earlier decoder-localization experiments identified repeated advanced indexing's
CPU backward accumulation as a sufficient source of early-update drift:
identical first predictions but slightly different gradients, amplified by
AdamW. A forward-equivalent indexing replacement and deterministic algorithms
both stabilized four synthetic and four real training updates, including
audit-hook/plain controls; see [FINDINGS_REPLAY.md](FINDINGS_REPLAY.md).

The full study establishes final-score variability with complete input matching,
and a reproducible deterministic trajectory for this workload. It does not
attribute every historical discrepancy solely to the decoder: the older
member-policy predecessor lacks full training-input hashes, and the full study
toggles deterministic algorithms rather than isolating only those two indexing
operations for 2,500 updates. The prior .09347 discrepancy remains a historical
contrast, not a number retrospectively decomposed by this test.

Same-seed/input matching is insufficient to call small retraining differences
exactly controlled. Future causal training comparisons should first establish
deterministic repeats for their environment and retain independent initialization
seeds. GPU behavior, other Torch versions, and other architectures are untested;
no production defaults have been changed.

Frozen-checkpoint predictions, parameter swaps, and query/support interventions
remain valid measurements of those exact models. In particular, the political
support-path result replicates across three separately initialized fixed models
per source; it does not interpret default execution repeats as a treatment
effect. See [FINDINGS_ROLE_CONTEXT.md](FINDINGS_ROLE_CONTEXT.md).

## Reproducibility and location

- Runtime `376f5af3`, isolated Tucker worktree
  `/dataMeR1/phil/gfm/prodigy-mechanisms-numfull`.
  `log/numerical_controls_training_20260906/verified/` and `evaluation/` are
  complete; the training/evaluation tmux session has exited.
- Compact receipts: `data/numerical_controls_verified/`; all metric exports:
  `data/numerical_controls_{original,fresh}/`.
- Rebuild with the `analyze_numerical_controls` module. Derived evidence:
  `data/numerical_controls_{cells,repeat_differences,full_with_historical}.csv`
  and `data/numerical_controls_validation.json`. Both earlier controls are
  retained descriptively, with full training-input verification distinguished.
- Three full-run analysis tests pass; the complete analysis suite now passes
  63 tests. Local branch/worktree: `codex/target-performance-mechanisms`,
  `/Users/philipp/projects/gfm/prodigy-mechanisms`. Private Git only, no public
  release, no user-job interruption.
