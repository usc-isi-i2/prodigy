# HK fixed-bank support selection

Run frozen-checkpoint inference in the isolated runtime worktree, using the
`prodigy` environment and an available owned GPU:

```bash
python -u scripts/experiments/setup/nm_hk_support_extremes/run.py \
  --device cuda:0 \
  --out /dataMeR1/phil/gfm/error_audit/nm_hk_support_extremes_20260908
```

The output directory must not already exist. Defaults reuse the canonical audit,
complete-input archive and HK mechanism cache. Override `--audit`, `--inputs`,
`--cache`, `--candidate-cap` or `--seed` as needed. No training is performed.
The trusted canonical torch archive is read metadata-first and only edge storages
are mapped; HK features come from the previously validated standalone artifact.

Each case uses at most 32 uniformly sampled eligible alternative support nodes.
One sampled training-edge context is fixed per candidate identity. Nearest and
farthest triples maximize/minimize mean learned cosine to the frozen query;
diverse starts nearest and greedily maximizes minimum cosine distance. Five
random triples use the same bank. The true support class is known: this is a
mechanistic diagnostic, not a deployable label-free selection algorithm.

Results: `../../analysis/evaluation/error_audit/FINDINGS_NM_HK_SUPPORT_EXTREMES.md`.
Private candidate IDs, sampled graphs, embeddings and logits remain on Tucker.
Aggregate locally with `summarize_nm_hk_support_extremes.py --input-root <results>`.
