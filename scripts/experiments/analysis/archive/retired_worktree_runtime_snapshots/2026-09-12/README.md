# Structured evidence from retired Tucker worktrees

This directory preserves 3,099 small structured artifacts (49 MB total) from 17
Tucker worktrees retired on 2026-09-12. Every JSON and JSONL record parses, every
CSV/TSV is readable, and no individual file exceeds the repository's 25 MB limit.

The source worktrees are `prodigy-archnative`, `prodigy-archtraj`,
`prodigy-boundary-repair`, `prodigy-centered-ridge-eval`,
`prodigy-centered-ridge-training`, `prodigy-classrefkv`, `prodigy-cls2500`,
`prodigy-degree-pairs`, `prodigy-directed-roles`, `prodigy-film8`,
`prodigy-follow-eval`, `prodigy-isolation-parity`, `prodigy-label-context`,
`prodigy-label-context-discovery`, `prodigy-ladder-xeval`, `prodigy-loosignal`,
and `prodigy-mechanisms-complement`.

The archive supplements, but does not replace, each study's canonical findings,
tables, and figures. It retains JSON, JSONL, CSV, TSV, PNG, and PDF evidence while
excluding checkpoints, tensors, W&B runs, caches, verbose worker/queue logs, and
ordinary source checkouts. Historical evidence restrictions still apply. In
particular, pre-2026-07-23 episodic static-link-prediction output is invalid, and
evaluation seeds must not be interpreted as resampled episode panels.
