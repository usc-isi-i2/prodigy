# Ukraine Suspended artifact repair — 2026-09-11

The canonical Tucker graph at `/dataMeR1/phil/data/ukr_rus_suspended/graphs/retweet_graph.pt` is replaced with the verified candidate from `/dataMeR1/phil/gfm/mixture-scaling/results/suspended_csv_repair_20260911_v2/graphs/retweet_graph_suspended.pt`. Both canonical metadata sidecars are updated from its metadata. The original graph and sidecars are preserved in `/dataMeR1/phil/data/ukr_rus_suspended/graphs.backup_before_csv_repair_20260911/`.

Two unquoted multiline bios corrupted CSV parsing and node/feature alignment. The repair reconstructs 56,440 source records in original node order, reuses bio embeddings by normalized-text content hash, and encodes two missing unique bios. The corrected graph has 56,440 nodes (previously 72,295), 354,209 unchanged edges, 768 feature coordinates and 56,440 labeled nodes. Verification reports no nonfinite features, identical edges, and 41,016 changed feature rows in the original node range. There are 12,283 all-zero feature rows. Detailed repair and verification JSONs remain beside the candidate artifact.

Original source CSV and embedding files are preserved for provenance; the catalog points to the corrected inputs used to build this graph. The candidate report's `candidate_not_promoted` field records its pre-promotion state; the canonical backup directory's promotion receipt records deployment.

## Consequences for experiments

This replacement affects future direct loads of the canonical path. It does not repair already trained models, cached graphs, or prior evaluation results. Audit/rebuild dataset-dependent caches in a new experiment directory before a corrected rerun; do not overwrite historical caches or infer compatibility from a matching filename. In particular, the September 11 MLP ladder rungs 6–9 trained with the old Suspended graph, and every Suspended-target evaluation used it. Their numerical results must remain marked as old-artifact results. Merged graphs embedding the old source also require separate rebuilding.

## Feature-distribution check

Eight Suspended-versus-other-graph comparisons were rerun using the repaired candidate, retaining the other graphs' samples. With 10,000 nonzero-feature nodes per graph and per-coordinate KS averaged over 768 dimensions, mean distance across the eight pairs falls from 0.2184 to 0.0732. The old apparent distribution outlier was largely an artifact defect. Node population changes mean this is not a matched-node causal decomposition.

Evidence: `/dataMeR1/phil/gfm/mixture-scaling/results/feature_dimension_ks_suspended_v2_s2026/results.json`; reproducible analysis in mixture-scaling branch `codex/feature-dimension-ks`, script `scripts/rerun_suspended_ks.py`.
