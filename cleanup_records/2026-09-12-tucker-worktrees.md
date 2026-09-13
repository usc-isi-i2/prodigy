# Tucker worktree retirement, 2026-09-12

This cleanup retained the main checkout, all September 8 or later paper/recovery
worktrees, `prodigy-nm-hk-goal-20260908`, `prodigy-proxy-a-seeds`,
`prodigy-vision-mixture-seeds`, and `prodigy-walk-mini-pilot`.

Before removal, each retired worktree was checked for running tmux sessions and
experiment processes, tracked modifications, untracked files, HEAD reachability,
canonical findings, committed result evidence, and missing figures. Redundant
runtime copies, checkpoints, W&B output, smoke artifacts, and controller logs were
not committed. The formerly unreachable `prodigy-vision-mixture-seeds` HEAD is
protected by `origin/codex/preserve-vision-mixture-seeds-20260912`.

Four studies required new preservation before retirement:

- training-role exposure: counts, crossed metrics, and qualified findings;
- social-to-FB15K adapter: eight canonical metric files and adapter-scoped findings;
- all-nine source-gradient diagnostic: the complete 200-step JSONL and a short,
  diagnostic-only disposition;
- strong MLP: the complete 25-cell result set, curves, protocol, and findings.
- PinSAGE final-core: the 16 corrected fixed-test cells, training receipts, and a
  ranking-versus-decision finding; earlier step-0 evaluator output was rejected.

Four setup/preflight-only worktrees (`msize`, `nm2sched`, `publickg-assets`, and
`publickg-evalprep`) have explicit non-result dispositions under
`scripts/experiments/analysis/archive/retired_worktree_dispositions/`.

Identifiable raw social-media examples and the matched-follow source sample were
not republished in Git. Five unique mechanism-example JSON files and
`original_sample.json` were copied to the restricted Tucker directory
`/dataMeR1/phil/data/worktree_retirement_20260912/`, JSON-parsed, and protected by
`SHA256SUMS`. The restricted archive totals 8.9 MB.

The current main graph catalog parses as JSON, but its existing catalog unit suite
has six failures involving the newer Cora, PubMed, Wiki, and FB15K-237 inventory
entries. The obsolete `prodigy-follow-matched` catalog edit was not retained: it
contained only 28 graphs versus main's 41 and would have removed 13 newer views.
