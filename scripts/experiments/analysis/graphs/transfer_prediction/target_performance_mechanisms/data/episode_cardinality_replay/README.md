# Complete episode-cardinality diagnostic

Tucker source:
`/dataMeR1/phil/gfm/prodigy-mechanisms-count/log/target_mechanisms/episode_cardinality_20260906`.
Frozen runtime `b882d95a`, launched 2026-09-06 12:36:34 UTC, completion verified by
12:38:50 UTC. CPU-only, eight threads, no new training or encoder passes.

This directory preserves all 630 metric cells, 90 checkpoint/prediction audits,
10 input inventories, protocol, completion receipt, and the complete per-batch/
head attention audit in `attention.json.gz`. Compression is lossless; the original
uncompressed SHA256 is
`6e67e0f34f3f756d05d686497778ccc773040dbf886f516d9701dd86a4bd2836`.
The compressed audit is intentionally tracked despite the general gzip ignore
rule. Raw prediction tensors remain in the Tucker run's `predictions/` subtree;
their exact paths and hashes are in `inventory.json`.

The local analysis accepts the compressed audit directly and recomputes complete
grid, identity, count and attention-conservation checks before testing the fixed
primary rule. All 2880 baseline batches are bit-exact for both saved post-metagraph
embeddings and full-model logits; unhooked restoration is also bit-exact. No
query outcomes were used to choose intervention weights.

The primary prediction failed for Facebook and TwiBot on both streams. All six
interventions are retained, including the modestly favorable reciprocal control.
Reweighted existing messages are not 30 distinct classes or new negative examples.
