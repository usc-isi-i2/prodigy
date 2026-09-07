# GILT component-compensation scope test

## Fixed scientific protocol (before crossover outcomes)

This is one mechanistic scope test, not an architecture leaderboard or a search
for a target on which the PRODIGY result repeats. GILT is selected because its
existing implementation has an explicit encoder/predictor boundary and compatible
saved checkpoints. VISION's interwoven label and graph processing makes the same
partition less clean. No GILT crossover outcomes informed this choice.

- Architecture: pinned official GILT components through this repository's
  `icl_arch_matrix.architecture_adapters.GILTAdapter`.
- Upstream: `/dataMeR1/phil/gfm/upstream/inductnode`, revision
  `ba46cf4ebd1931712854708c221eaba646641785`.
- Source: `covid_political`, training seed 0, native-source classification episodes.
- Checkpoints: updates 100 and 900, fixed in advance, not target-selected.
- Checkpoint root on Tucker:
  `/dataMeR1/phil/gfm/prodigy-archnative/state/icl_arch_native_source_900_seed0/gilt/ss_covid_political_s0/checkpoint/`.
- Target: `twibot20` only; 128 fixed, paired binary 10-shot episodes through the
  existing shared evaluation protocol. Record the actual episode fingerprints.
- No training, fine-tuning, alignment adapter, checkpoint search, or target-label
  model selection. Use owned GPU 3 only after checking it is idle.

## Conditions and predictions

Cross the two encoder states with the two predictor states. State ownership must
be exhaustive: `encoder.*`, `predictor.*`, and a fixed `feature_projection` that
must be identical across checkpoints. Recompute support prototypes from the chosen
encoder's features, retaining the same labels and query ordering. Use evaluation
mode throughout and verify the matched conditions against unchanged native
adapter forwards on exactly the same episodes.

The prospective prediction is the same component direction replicated in PRODIGY:

1. Later predictor improves performance with either encoder.
2. Later encoder reduces performance with either predictor.
3. Earlier encoder plus later predictor beats both native endpoints.

Report accuracy, macro-F1, ROC-AUC, and NLL for every condition and paired effects.
Accuracy is the primary directional test; report metric disagreement rather than
selecting a favorable measure. Include a fixed support-centered intermediate ridge
readout (lambda 1, row-L2 normalization, no intercept) and the same rule on raw
features. These prevent a crossed model's gain over native endpoints from being
mistaken for superiority over a strong refitted readout. Query labels are for
scoring only.

## Interpretation and stopping rule

A matching direction would extend the component observation to a second learned
in-context model family under this source/target protocol. It would not identify
a universal geometric mechanism or validate a checkpoint-selection algorithm.
A failure is a boundary on generality; do not replace the chosen target, checkpoint
pair, or architecture after seeing results to obtain a favorable replication.

This is not a controlled architecture ablation: GILT differs in source objective,
update budget, projected dimension, symmetrized graph processing, and normalization.
Its encoder uses simplified propagation with trainable normalization rather than
PRODIGY's full learned message passing. These differences must be disclosed.
The checkpoints are from our source-confined native classification protocol, not
the full multitask pretraining recipe in the GILT paper.

## Execution status

Runner partition and failure-path tests pass locally. The actual predictor
signature and cache metadata were checked read-only on Tucker. The fixed input
fingerprint is `c52eedbda4ef0539387be507588f0477560fc57987270d62405daeda5fe6d92d`.

From the dedicated Tucker checkout, with the `prodigy` environment active:

```bash
python -m scripts.experiments.setup.gilt_component_crossover.run \
  --checkpoint-dir /dataMeR1/phil/gfm/prodigy-archnative/state/icl_arch_native_source_900_seed0/gilt/ss_covid_political_s0/checkpoint \
  --cache /dataMeR1/phil/gfm/prodigy-encoder-solver-isolation/log/isolation_eval_20260907/original/twibot20/twibot20 \
  --upstream /dataMeR1/phil/gfm/upstream/inductnode \
  --output /dataMeR1/phil/gfm/prodigy-gilt-component/log/gilt_component_crossover_20260907 \
  --gpu 3
```

This prints the plan and input hashes without evaluating. Append `--execute` only
after dry-run validation and an idle-GPU check. An existing output is never
overwritten. `execution_status.json` is the completion authority; inspect it
before using any partial outputs. The runner also reports global-positive binary
F1, without replacing the nominated macro-F1 measure.
