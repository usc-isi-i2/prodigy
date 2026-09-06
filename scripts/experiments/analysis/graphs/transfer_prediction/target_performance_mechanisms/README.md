# Target-performance mechanisms

Start with [FINDINGS_REPLAY.md](FINDINGS_REPLAY.md). Setup and operational
instructions are in `scripts/experiments/setup/target_performance_mechanisms/`.
The [summary figure](figures/target_bottlenecks.png) contrasts target-input signal
and full-model performance on the original and fresh episode streams.

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
