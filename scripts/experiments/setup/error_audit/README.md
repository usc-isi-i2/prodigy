# Correct/incorrect prediction audit

This experiment exports prediction-level evidence for qualitative diagnosis.  It
does not train new models.  Every raw row retains the model prediction, ground
truth, correctness/error, episode provenance, and three node ids from the exact
sampled subgraph used by the encoder.

## Questions

1. Which profile and neighbourhood regimes produce correct versus incorrect NM,
   node-classification, node-regression, and link-prediction outputs?
2. Are apparent failures concentrated in missing-bio, low-degree, cross-community,
   class-specific, or target-tail cases?
3. Do high-confidence failures have qualitatively different context from matched
   correct cases?

## Protocol

- Use one explicit frozen checkpoint list and record its git revision.
- NM and classification use the existing fixed split-derived episode stream.
  Compare models on the same dataset/task/shot configuration.
- Regression uses the repaired per-episode StandardScaler + Ridge probe.  It exports
  all continuous predictions; analysis calls the bottom and top absolute-error
  quintiles `low_error` and `high_error`, rather than pretending regression has an
  exact correct class.
- Static LP uses the repaired two-endpoint cosine evaluator and degree-matched
  negatives.  Score orientation and the balanced-accuracy decision threshold are
  fitted on validation pairs only, then locked for test-pair correct/incorrect calls.
- Temporal LP is excluded.  Its old episodic evaluator is endpoint-blind and it has
  not yet received a valid directed temporal replacement.
- The raw export contains both correct and incorrect records.  Report sampling is
  balanced and never substitutes for aggregate metrics.

The graph catalog does not provide all tasks on one graph.  A typical pilot uses
Midterm for NM/regression/static LP and `covid_political` for classification.

## Tucker execution

Create a model list outside git, then run in the experiment's dedicated worktree:

```bash
export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

export MODEL_LIST=/dataMeR1/phil/gfm/error_audit/model_list.txt
bash scripts/experiments/setup/error_audit/run_episode_audit_tucker.sh
bash scripts/experiments/setup/error_audit/run_regression_audit_tucker.sh
bash scripts/experiments/setup/error_audit/run_static_lp_audit_tucker.sh
```

Use a dedicated worktree and check `tmux ls` before pulling or changing revisions.
For a detached run, put the PATH export inside the tmux command as documented in
the repository `AGENTS.md`.

## Outputs

- NM/CLS: `log/<eval-run>/data/predictions_{val,test}_step0.jsonl`
- REG: `<OUT_DIR>/<dataset>__reg_probe_examples.jsonl`
- static LP: `<OUT_DIR>/<dataset>__pair_lp_examples.jsonl`

Keep these under `/dataMeR1`; they may contain recoverable user identifiers and the
analysis step adds raw profile bios.  Do not commit them.

See `scripts/experiments/analysis/evaluation/error_audit/README.md` for enrichment and report
generation.

Local protocol gates (use the local `prodigy` environment) are collected in
`run_gate.sh`.
