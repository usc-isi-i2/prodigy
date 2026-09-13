# PublicKG episode-cache reproducibility snapshots

This directory preserves the compact metadata and aggregate outputs from five
PublicKG experiments whose materialized `paired_episodes/*.pt` caches were removed
during the 2026-09-12 Tucker storage cleanup.

The removed files duplicated sampled graph node and edge embeddings in every
episode and occupied about 139 GiB. They were replay caches, not the only copy of
the source dataset or model checkpoints.

Each run snapshot retains its protocol, effective parameters, model and import
contracts, sampler stream information where applicable, completion status,
aggregate evaluation metrics, ordinary logs, and the original episode checksum
index. The checksum index documents the deleted cache but cannot validate it after
deletion.

Scientific reruns require the recorded source dataset, upstream code, environment,
and checkpoint. Their paths and hashes are recorded in `protocol.json`. At cleanup
time both seed-0 and seed-1 checkpoints, the public dataset, and upstream evaluation
code were present on Tucker. GPU and library nondeterminism may prevent regenerated
`.pt` archives from being byte-identical even when aggregate conclusions reproduce.

Source Codex task: `correlate target losses with source graph (2)`
(`01a07a95-2d16-71f1-a334-3f4e6bce1517`).
