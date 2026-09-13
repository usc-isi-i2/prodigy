# Retired PublicKG worktree runtime snapshot

This directory preserves the non-tensor runtime outputs that remained in the
`prodigy-publickg-paired-state` worktree immediately before its removal on
2026-09-12. The worktree HEAD (`0e68737d`) was already an ancestor of main.

Materialized `.pt` episode and batch caches are intentionally excluded. Those
files repeatedly stored sampled graph features and can be regenerated from the
protocols, source data, and checkpoints. JSON protocols, completion records,
aggregate summaries, CSV audits, text logs, and W&B metadata are retained here.

The canonical conclusions remain in `../../FINDINGS.md`. This snapshot exists to
retain detailed provenance and secondary analysis inputs without keeping several
gigabytes of redundant replay tensors.
