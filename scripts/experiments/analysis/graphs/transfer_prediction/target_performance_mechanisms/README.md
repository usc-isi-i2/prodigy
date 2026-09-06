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

The member-policy experiment uses the completed-run verifier's exact
`arms.json` and `DONE.json` under `data/member_training_verified/`, plus
`data/member_replay_{original,fresh}/`. The analysis refuses smoke or incomplete
results and requires all 24 models, all five targets, both streams, and matching
verified terminal-weight hashes. Run it as a module because its validator shares
the trajectory module's decoder contract:

```bash
MPLBACKEND=Agg /opt/homebrew/bin/python3.11 -m \
  scripts.experiments.analysis.graphs.transfer_prediction.target_performance_mechanisms.analyze_member_intervention
```

It reports every training seed's Hong Kong-minus-Ukraine retention effect on the
prespecified COVID Political/Facebook/TwiBot primary panel, with role effects,
interactions, all-target outcomes, and actual consumed exposure as secondary
outputs. No significance test treats targets, episode streams, checkpoints, or
mixture partners as independent training-seed replications.
