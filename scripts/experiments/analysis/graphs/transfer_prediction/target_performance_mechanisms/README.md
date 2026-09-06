# Target-performance mechanisms

Start with [FINDINGS_REPLAY.md](FINDINGS_REPLAY.md). Setup and operational
instructions are in `scripts/experiments/setup/target_performance_mechanisms/`.

Reproduce the currently completed three-target summary:

```bash
MPLBACKEND=Agg /opt/homebrew/bin/python3.11 \
  scripts/experiments/analysis/graphs/transfer_prediction/target_performance_mechanisms/analyze_replay.py \
  --targets covid_political,facebook_page_reference,election2020
```

The raw per-target JSONL files in `data/replay/` are copied from each target's
`metrics.jsonl` in Tucker's initial replay run. They contain 250 rows each.
Large cached batches, query logits, and latent tensors stay on Tucker. All paths,
revisions, and the distinction between completed targets and incomplete sweeps
are recorded in the findings. `data/source_sampler_summary.json` and
`data/source_sampler_protocol.json` are the exact source-audit exports.

The preceding lattice-only audit remains in the original main local worktree:
`/Users/philipp/projects/gfm/prodigy/scripts/experiments/analysis/graphs/transfer_prediction/target_performance_mechanisms/`.
It is not replaced by these experiments. The broader research protocol lives
outside git at `../paper/planning/target_performance_research_design_2026-09-06.md`.
