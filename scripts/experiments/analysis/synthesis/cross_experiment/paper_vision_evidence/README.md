# Paper vision evidence

Start with [FINDINGS.md](FINDINGS.md). This is a local, read-only-source synthesis
for the RQ1/RQ2 metrics in `SGFM Paper Presentation(1).pptx`; it does not launch
training or evaluation. Existing evidence is kept in separate protocol panels.

Rebuild from the repository root:

```bash
MPLBACKEND=Agg /opt/homebrew/bin/python3.11 scripts/experiments/analysis/synthesis/cross_experiment/paper_vision_evidence/assemble.py
```

Dependencies: pandas and numpy. The script reads the five completed source
exports listed in `data/validation.json`, plus the source-order plan and graph
catalog. It checks coverage, keys, metric bounds, fixed checkpoint endpoints,
paired fingerprints, seed counts and agreement with existing lattice/schedule
analyses. It overwrites only this analysis leaf's generated files.

For each individual mixture, use `data/model_metrics.csv`. For slide-level
order/rung summaries use `data/deck_metrics.csv`. Use `common4_*` scopes when
comparing directions across models; SAMGPT does not share the same Facebook
classification task. Neither this scope nor shared target names makes the
different native protocols a controlled architecture comparison.

`data/downstream_cells.csv` preserves physical IDs and training seeds.
`data/paired_gains.csv` additionally records baseline values and physical IDs;
schedule model IDs in aggregate tables omit the seed suffix so that training
seeds are averaged, not counted as additional targets. Logical order views can
reuse a physical checkpoint. Do not pool those views as independent runs.

Source CSVs, including shared accumulated result tables, are never rewritten.
Missing results are documented in FINDINGS rather than represented as zeros.

Created in worktree `/Users/philipp/projects/gfm/prodigy`, branch `main`.
The generation revision and source hashes are in `data/validation.json`.
