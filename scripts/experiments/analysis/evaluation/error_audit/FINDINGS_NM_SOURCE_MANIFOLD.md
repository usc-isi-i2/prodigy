# NM source-manifold affinity on fixed HK and Ukraine inputs

9 September 2026. This diagnostic asks whether native versus foreign NM
outcomes are explained by where a query's sampled subgraph lies in the learned
pre-metagraph representation space. It reuses the same 61,440 canonical query
occurrences per target and the frozen HK and Ukraine checkpoints. No graph was
loaded, no embedding was recomputed, and no GPU was used.

## Protocol

Within each checkpoint's own representation space, form balanced 256-node HK
and Ukraine reference banks. Nodes are selected by deterministic SHA256 order,
without outcomes; five independent bank selections test sensitivity. Each bank
uses the first cached occurrence of a selected node. For a target query, average
its top-five cosine similarities to each bank. Exact node identity is excluded
only from its own graph's bank because graph-local integer IDs are not shared
identities across graphs.

Three scores are audited: similarity to the target graph bank, similarity to
the other graph bank, and their difference. AUC measures whether each score
ranks that model's correct occurrences above its failures. The node-balanced
AUC gives every query identity total weight one across its repeated
occurrences. Reference replicates share the evaluation queries and are
sensitivity checks, not five independent experiments.

## Result

| Target | Model | Score | Occurrence AUC, mean [range] | Node-balanced AUC, mean [range] |
|---|---|---|---:|---:|
| HK | HK native | target similarity | .560 [.542, .583] | .562 [.542, .571] |
| HK | HK native | other similarity | .602 [.591, .607] | .656 [.647, .668] |
| HK | HK native | target − other | .433 [.404, .478] | .386 [.378, .397] |
| HK | Ukraine foreign | target similarity | .431 [.425, .434] | .386 [.383, .390] |
| HK | Ukraine foreign | other similarity | .502 [.419, .574] | .535 [.500, .556] |
| HK | Ukraine foreign | target − other | .421 [.373, .471] | .360 [.337, .376] |
| Ukraine | Ukraine native | target similarity | .394 [.367, .404] | .398 [.389, .409] |
| Ukraine | Ukraine native | other similarity | .482 [.463, .508] | .469 [.456, .490] |
| Ukraine | Ukraine native | target − other | .475 [.455, .496] | .487 [.465, .503] |
| Ukraine | HK foreign | target similarity | .480 [.464, .494] | .491 [.479, .505] |
| Ukraine | HK foreign | other similarity | .596 [.548, .643] | .598 [.567, .632] |
| Ukraine | HK foreign | target − other | .368 [.327, .407] | .372 [.344, .396] |

The strong version of the source-coverage hypothesis fails this test. Native
success is not consistently associated with greater target-manifold affinity.
Only HK's absolute similarity under the native HK encoder is weakly positive;
similarity to the Ukraine bank predicts HK success more strongly. Under the
native Ukraine encoder, correct Ukraine queries are *less* similar to the
Ukraine bank on average. Target-minus-other affinity is at chance or points in
the wrong direction in every cell. Equal node weighting preserves these
directions, so frequent HK queries do not create the result.

The reference-bank score is likely mixing representation density, anisotropy,
and generic query difficulty. High affinity can identify a common graph region
without separating the correct candidate inside an episode. The result thus
fits the existing stage evidence: source training matters, but global graph
identity or nearest-manifold proximity is not the missing decision variable.

## Claim boundary and next test

This is a controlled-input correlational diagnostic. It rejects nearest-neighbor
affinity to held-out source nodes as a sufficient episode-level account. It
does not measure density under the historical training stream, prove that true
out-of-support regions are harmless, or isolate topology from features. The
reference embeddings are sampled contexts, and their first occurrences are one
realization of those contexts.

Loading actual training-node banks is not justified as the immediate next
step: the cheaper held-out-manifold prerequisite did not show the predicted
cross-target pattern. The subsequent [episode-relative margin audit](FINDINGS_NM_SOURCE_MARGIN_STAGES.md)
therefore compares native and foreign true-versus-best-rival separation before
and after the metagraph. It finds that Ukraine's native advantage is mostly
pre-metagraph, while HK's is mostly created by the support-conditioned readout.

Compact aggregate: [nm_source_manifold_v4_aggregate.json](data/canonical_split/nm_source_manifold_v4_aggregate.json).
Private per-occurrence rows and five receipts are under
`/dataMeR1/phil/gfm/error_audit/nm_source_manifold_v4_rep{0..4}`. The five runs
took about one minute total on two low-priority CPU threads. Runtime branch
`codex/nm-hk-goal-20260908`, revision `433f33ec`; no GPU was allocated.
