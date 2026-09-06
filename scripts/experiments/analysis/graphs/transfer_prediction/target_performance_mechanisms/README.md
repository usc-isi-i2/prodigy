# Target-performance mechanisms

Start with [FINDINGS_REPLAY.md](FINDINGS_REPLAY.md). Setup and operational
instructions are in `scripts/experiments/setup/target_performance_mechanisms/`.
The [summary figure](figures/target_bottlenecks.png) contrasts target-input signal
and full-model performance on the original and fresh episode streams.

Latest: [FINDINGS_NATURAL_SUPPORT.md](FINDINGS_NATURAL_SUPPORT.md) tests natural,
unmodified support replacements on all six Ukraine/Hong Kong seed controls,
five targets and both streams. Both declared political predictions pass, but
support-only selection is not a general remedy: it usually worsens Ukraine's
political NLL and is less effective than eight-context averaging for that loss.
Rebuild the independent validation and figure with the `analyze_natural_support`
and `plot_natural_support` modules. The larger labeled pool is not a fair
standard 10-shot evaluation.

[FINDINGS_ROLE_CONTEXT.md](FINDINGS_ROLE_CONTEXT.md) quantifies the
individual-case hypotheses across all nine historical singletons, five targets,
two streams, and eight query/support conditions (720 cells). A separate complete
three-seed check replicates Hong Kong's political support-edge penalty and traces
it exactly through changed label representations, with query vectors unchanged.
Run `analyze_role_context --input <data/role_context_replay>` and
`analyze_support_path` as modules to independently validate and rebuild results.
The Facebook anecdotal rescues do not become a general context-removal benefit.

For actual individual examples, read [FINDINGS_EXAMPLES.md](FINDINGS_EXAMPLES.md):
24 outcome-stratified cases, verified raw text/feature identities, and 360
query-only versus support-only interventions with bit-exact baseline replay.
Numeric evidence is `data/individual_examples.json`; bulk profile text remains
in the private, unversioned paper-evidence folder identified in that file.
`summarize_examples.py --input <private-text-records> --output <new-json>` rebuilds
the compact evidence. These selected cases are not prevalence estimates.

`analyze_update_numerics.py` validates the complete fixed-input decoder-localization
receipts in `data/control_{decoder,real_input}_numerics.json`. It checks all
12 replays per workload, initialization/input hashes, mode identities, decoder
forward parity, and agreement between tensor digests and reported differences.
Run as a module with `--receipts <synthetic-receipt> <real-input-receipt>` to rebuild
`data/control_numerics_{steps,summary}.csv` and the validation JSON. These are CPU
reproducibility diagnostics, not target-performance results or extra seeds.

The prespecified full-run stress-case analysis is `analyze_numerical_controls.py`.
It requires the complete four-control `data/numerical_controls_verified/` receipts,
including state and logit comparisons, plus both `data/numerical_controls_{original,fresh}/`
replay exports. It reports every repeat contrast and the two earlier controls
without counting same-seed executions as independent seeds. A failed determinism
prediction is retained as an outcome; incomplete or input-mismatched runs fail
the validity gate. The full-run stress test is now complete:
[FINDINGS_NUMERICS.md](FINDINGS_NUMERICS.md). Default matched-input repeats differ
by up to .0719 full-model AUC; deterministic saved states and predictions match
exactly across all five targets and both streams. This is one selected stress
case, not a typical-variability estimate or a transfer-performance remedy.

Reproduce the completed five-target summary:

```bash
MPLBACKEND=Agg /opt/homebrew/bin/python3.11 \
  scripts/experiments/analysis/graphs/transfer_prediction/target_performance_mechanisms/analyze_replay.py
MPLBACKEND=Agg /opt/homebrew/bin/python3.11 \
  scripts/experiments/analysis/graphs/transfer_prediction/target_performance_mechanisms/analyze_pairs.py
MPLBACKEND=Agg /opt/homebrew/bin/python3.11 \
  scripts/experiments/analysis/graphs/transfer_prediction/target_performance_mechanisms/analyze_pairs.py --fresh
MPLBACKEND=Agg /opt/homebrew/bin/python3.11 \
  scripts/experiments/analysis/graphs/transfer_prediction/target_performance_mechanisms/analyze_fresh_and_coverage.py
MPLBACKEND=Agg /opt/homebrew/bin/python3.11 \
  scripts/experiments/analysis/graphs/transfer_prediction/target_performance_mechanisms/analyze_fresh_interventions.py
MPLBACKEND=Agg /opt/homebrew/bin/python3.11 \
  scripts/experiments/analysis/graphs/transfer_prediction/target_performance_mechanisms/plot_target_bottlenecks.py
```

The raw per-target JSONL files in `data/replay/` are copied from each target's
`metrics.jsonl` in Tucker's initial replay run (the first three targets) or its
completed continuation (TwiBot and Ukraine Suspended). They contain 250 rows each.
Large cached batches, query logits, and latent tensors stay on Tucker. All paths,
revisions, and the distinction between completed targets and incomplete sweeps
are recorded in the findings. `data/source_sampler_summary.json` and
`data/source_sampler_protocol.json` are the exact source-audit exports.

The preceding lattice-only audit remains in the original main local worktree:
`/Users/philipp/projects/gfm/prodigy/scripts/experiments/analysis/graphs/transfer_prediction/target_performance_mechanisms/`.
It is not replaced by these experiments. The broader research protocol lives
outside git at `../paper/planning/target_performance_research_design_2026-09-06.md`.

## Prespecified follow-up analyses

`analyze_trajectories.py` validates all nine specialist trajectories at updates
100/300/900/2500 on both episode streams. It requires complete
`data/trajectory_{original,fresh}/` exports and matches terminal weights, metrics,
and episode fingerprints to the established replay. It reports endpoint changes
and complete curves; it does not select the best target checkpoint.
`plot_trajectories.py` renders the all-source endpoint-change summary.

The completed post-hoc TwiBot cue check uses saved prediction-only exports, not
new model fitting. `analyze_cue_alignment.py --input <completed-output>` validates
the checkpoint/input provenance and summarizes all four checkpoints against the
support-only incoming-degree, raw-center, and raw-context probes. Compact results
are `data/twibot_cue_alignment_*.csv`. The raw tensor exports remain in ignored
runtime storage. The executable comparison is
`scripts/experiments/setup/target_performance_mechanisms/analyze_twibot_cue_alignment.py`.

The member-policy experiment uses the completed-run verifier's exact
`arms.json` and `DONE.json` under `data/member_training_verified/`, plus
`data/member_replay_{original,fresh}/`. The analysis refuses smoke or incomplete
results and requires all 24 models, all five targets, both streams, and matching
verified terminal-weight hashes. Run it as a module because its validator shares
the trajectory module's decoder contract:

```bash
MPLBACKEND=Agg /opt/homebrew/bin/python3.11 -m \
  scripts.experiments.analysis.graphs.transfer_prediction.target_performance_mechanisms.audit_member_contract
MPLBACKEND=Agg /opt/homebrew/bin/python3.11 -m \
  scripts.experiments.analysis.graphs.transfer_prediction.target_performance_mechanisms.analyze_member_intervention
```

It reports every training seed's Hong Kong-minus-Ukraine retention effect on the
prespecified COVID Political/Facebook/TwiBot primary panel, with role effects,
interactions, all-target outcomes, and actual consumed exposure as secondary
outputs. No significance test treats targets, episode streams, checkpoints, or
mixture partners as independent training-seed replications.

The independent contract audit also checks all 24 saved effective configurations
against their declared CPU recipes and common non-treatment parameters, and maps
the consumed episode-source labels to the exact graph audit's source-name order.
Those labels are derived by the production collator from all member centers'
graph IDs. This is not a second independent lookup of every ID in the full graph.
`--config-only` is allowed before training finishes and does not publish the
completed contract receipt. The compact configuration inventory reconstructs
the complete original dictionaries from one shared base plus lossless deltas.
After the complete primary analysis, `plot_member_intervention.py` displays the
observed repeated-member fraction, each source's primary-panel retention effect,
and the prespecified Hong Kong-minus-Ukraine contrast, retaining every seed and
both episode streams. It does not select a target, seed, or checkpoint.

`analyze_label_interface.py` is the fail-closed analysis for the completed frozen
label-vector controls. It requires all nine specialists, all six conditions,
all five targets, both episode streams, exact input/weight matches, and successful
norm/flag audits. Run as a module after the completed outputs and input receipt
are collected. These graph runs use class-keyed deterministic random label
vectors, not semantic label-text embeddings; legacy control names are unchanged.

`analyze_campaign_cls.py --nm-results <original-nm-results.csv>` validates the
declared 15-run, 30-checkpoint cross-task replay. Common 6,000-update and original
source-validation-selected comparisons are kept separate. Selected-checkpoint
NM comparisons require the original checkpoint file hashes to match. The manifest
and explicit forward-incompatible exclusions live under setup's
`data/campaign_cls_manifest_v2/`. Only TwiBot is a held-out source; one seed and
repeat conditions must not be counted as independent training replications.
The replay is complete. `plot_campaign_cls.py` renders its two checkpoint rules
and matched selected-checkpoint NM outcomes in `figures/campaign_nm_cls_comparison.png`.

`analyze_member_cue_alignment.py` reports the prospective exploratory secondary
cue effects only after the complete primary member-policy analysis passes.
Undefined correlations from constant predictions remain missing; no arm is
dropped to compute a more favorable contrast.

`analyze_member_initial_reference.py` is the completed exploratory supplementary
comparison to the three exact saved step-zero states, each shared by eight
source/policy models. Run after the full primary analysis and collecting
`data/member_initial_{original,fresh}/` plus `data/member_initial_input_validation.json`.
It checks all initial weight digests against consumed-training records and all
320 batch hashes against the terminal reference, then reports all 4,080 same-seed
initial-to-terminal changes. This is not an arbitrary new random baseline or a
retroactive initial reference for the separate historical nine-source trajectories.
The primary contrast is unchanged. Four additional tests cover exact seed/weight,
complete-grid, finite-metric, and cached-input matching.

The exploratory exact-readout-weight swaps are analyzed by
`analyze_readout_interventions.py`, after the completed primary and exact-initial
analyses above. It requires all 48 hybrids on five targets, 17 decoders, and two
streams (8,160 rows), the exact saved-state manifest, and the separate tensor/
input/upstream-output audit under `data/readout_audit/`. Reference metrics come
from the same training seed's verified terminal or exact initial state, depending
on swap direction. Raw and pre-readout probes must remain unchanged. Both direct
swap effects and background-by-readout interactions retain every policy and seed;
the rest-of-network background is not described as an encoder-only intervention.
The secondary cue analysis checks all 1,800 hybrid/background cells and reproduces
the existing 432 terminal cue cells before calculating paired changes. It does
not fit query labels or equate a probe improvement with a full-model remedy.
The additional center-branch cue was declared before reading swap outcomes, after
the existing exact-initialization table showed center features already declining
while pooled features improved. The original three cue stages are retained. The
576 initial-to-terminal cue differences separate these parallel encoder branches.

The full swap replay and exact-output audit are complete. After reproducing the
analysis, `plot_readout_interventions.py` shows representation-probe versus
full-model changes for restoration and reverse implantation. Both figures keep
every policy/seed and both episode streams; the complete tables retain all five
targets. A probe improvement is not presented as an end-to-end improvement.

The prospective training follow-up has a separate complete-grid analysis,
`analyze_readout_training.py`. It requires 18 new free/frozen-readout training
runs (Ukraine/Hong Kong/COVID, seeds 0/1/2), all saved checkpoints and per-update
constraint checks, complete paired training-input fingerprints, both fixed test
streams, and every target/decoder. The primary endpoint is frozen-minus-free
**Facebook full-model AUC**, averaged over the three sources within each seed.
It separately reports each source/seed, all-target probe/model effects, and
reproduction differences for the six historical standard-policy controls. The
positive-consistency criterion was fixed before outcomes; probe-only improvement
does not meet it. This is an exploratory follow-up, not a fresh-domain test.

The separate historical mixture diagnostic uses the frozen 54-model lattice in
`data/mixture_complementarity_inputs/`, not the sampler-corrected models. The
Tucker logit export/analysis is complete; `analyze_mixture_complementarity.py`
requires all 540 model cells, 450 mixture comparisons, 1,350 error strata and
10 input sets under `data/mixture_complementarity_predictions/`. It independently
checks source coverage, original aggregate parity, metric arithmetic, binary
ensemble unanimity, error-stratum accounting and shared checkpoint/input identity.
It reports all/foreign/target-seen panels for pairs and LOO separately, retaining
both episode streams. Equal-probability averaging is primary; equal-logit is
secondary. Within-target rank associations adjust for constituent mean AUC and
AUC range, but do not provide independent-pair inference: pairs share sources,
and there is only one training seed. Undefined partial associations remain
undefined. Ensembles use 2x/8x the training updates and model forwards, not a
measured wall-clock multiplier; this is not causal interference evidence. Toy
arithmetic and mutation tests exercise these reporting gates. After validation,
`plot_mixture_complementarity.py` renders all 290 foreign cases across both
streams; the verified results are in `FINDINGS_REPLAY.md`.

`analyze_corrected_sampler.py` separately checks whether the historical stage/
target patterns persist in the nine newer corrected-sampler singleton runs.
It requires all 36 verified saved checkpoints and all 6,120 replay cells, with
the complete immutable-input receipt under `data/corrected_sampler_verified/`
and metrics in `data/corrected_sampler_{original,fresh}/`. Comparisons join the
historical trajectory by source, update, target, decoder and episode stream—not
by selecting a best checkpoint. Both all-source and foreign-source summaries are
retained. This is a joint retention/role/RNG recipe comparison on one training
seed, not a matched-training or role-only causal intervention. Four tests cover
the full grid, fixed endpoint arithmetic, source/budget metadata, raw-probe/input
identity, missing/mutated artifacts, and complete plotted trajectories. This
replay is complete; `plot_corrected_sampler.py` shows every source and saved
checkpoint for the four TwiBot diagnostic branches, with both episode streams.

`analyze_mixture_budget.py` extends only the historical complementarity analysis
to fixed specialist checkpoints at 100/300/900/2500 updates. It requires all
810 prediction cells, 1800 comparisons and 5400 error strata under
`data/mixture_budget_predictions/`, exact agreement with the saved trajectory
weights/inputs, and full reproduction of every earlier step-2500 result. The
budget rule selects the largest available specialist step with K × step ≤ 2500:
900 for pairs and 300 for LOO. All steps, targets and source-membership panels
remain visible. The primary exploratory prediction is that the foreign Facebook
LOO ensemble advantage remains positive on both streams at step 300; a failure
must be reported. These are update/episode budgets, not matched training FLOPs,
capacity, inference cost or training data. No query-selected ensemble weights.
The complete analysis is now verified; its primary Facebook prediction failed.
`plot_mixture_budget.py` renders every foreign case at all four saved steps,
including the fixed budget rule, inference-cost caveat and both episode streams.
