# NM failure hierarchy: from one HK query to source-dependent transfer

9 September 2026. This synthesis separates what the canonical episode audits
establish from what the transfer matrices only correlate. It includes the
completed matched HK training intervention at its fixed terminal checkpoint.

## One query and one episode

The first canonical HK failure is a genuine candidate-set error. Its query has
test edges to classes 0, 27, and 29, but the native model chooses class 7, which
has no recorded train, validation, or test edge. Raw center cosine, raw sampled
neighborhood mean, learned pre-metagraph similarity, and the final model choose
four different rivals. The encoder therefore changes the geometry materially.

The query has the greatest sampled-node overlap with its assigned class, but
the learned positive-support reference favors class 7. Removing class 7's
positive messages changes the winner; removing negative messages does not make
a legal class win. This is controlled evidence for the immediate decision in
this query, not an explanation of why training learned it.

Its episode is difficult without collapsing: 18/120 queries are correct and 26
of 30 classes are predicted. Positive-support terms favor the wrong winner in
99/102 errors and are the largest pro-error component in 88/102. The episode
therefore exhibits many class-reference errors rather than one dominant-label
failure.

## Full canonical HK population

The fixed benchmark contains 61,440 query occurrences over 512 episodes.
Assigned-anchor accuracy is 17.86%. Multi-positive scoring raises it to 23.26%,
proving that 3,315/50,464 assigned errors (6.57%) are evaluation artifacts in
which the prediction is another held-out neighbor. Ambiguity is common—28,407
occurrences have multiple valid test anchors—but 21,332 ambiguous occurrences
remain wrong even when every test-positive candidate counts. Among 33,033
uniquely answerable occurrences, 25,817 are wrong. Evaluation ambiguity is a
real correction, not the main explanation of low HK accuracy.

The positive-reference localization scales to the complete population. It
favors the wrong winner in 47,130/50,464 native errors (93.39%) and is the
largest rival-favoring component in 77.76%. Correct controls reverse direction:
it favors truth in 82.02% of 10,976 successes. The pattern remains in uniquely
answerable errors (92.28%), errors under multi-positive scoring (93.29%), every
episode, and node-weighted repeated-query summaries. This localizes the final
decision pathway with high confidence; it does not make 93.39% a causal
training-explanation fraction because the additive terms share learned
attention and normalization.

Other evidence remains conditional or correlational. Rival support subgraphs
contain the query center in 60.93% of uniquely answerable errors versus 41.30%
of successes, and failed queries have higher degree and recurrence. Yet the
first query has useful overlap with truth, and inference-time deletion of
training-edge-contradicted negative messages slightly hurts the full benchmark.
Sample overlap, degree, and recurrence identify difficult populations but do
not independently cause the errors demonstrated so far.

## Native and foreign models on identical inputs

On the same HK episodes, HK versus Ukraine accuracy is 12.98% versus 10.88%
before the metagraph and 17.86% versus 11.68% after it. HK's native advantage is
small in the learned query/support geometry and is amplified by the readout.
The two models agree on a prediction only 17.32% of the time, and shared failure
usually does not mean the same distractor.

On the same Ukraine episodes, the stage boundary differs: Ukraine versus HK is
38.89% versus 17.05% before the metagraph and 44.15% versus 18.15% finally.
Ukraine's native advantage is already mostly representational. Positive-support
dominance appears in roughly 92% of errors for both models on both targets, so
it is the common NM computation pathway rather than an HK-specific defect.
Source training can place the important divergence before the metagraph or in
the final support-derived decision depending on the target.

## Training conflict and the unresolved causal link

Complete realized 2,500-batch training streams show that an HK query also
appears as a differently labelled support in 22.20–22.23% of occurrences;
59.28–59.30% of support identities receive multiple support labels, and every
HK episode contains a query/support identity conflict. The corresponding
query/support rates are 2.10% for Ukraine and 0.89% for COVID. This establishes
source-specific exposure under the historical lowest-ID recipe.

A no-update intervention making duplicate support identities carry consistent
soft labels changes HK's full-parameter gradient substantially (mean relative
distance .516) but worsens absolute trained-state NM loss. It proves that label
binding changes the learning signal, not that it explains final errors or
repairs them. The matched overlap-aware training pair is the first narrow test
of whether actual training-adjacent rival supports account for part of HK's
missing separation. Its query targets, centers, members, sampled contexts,
initialization, and optimizer schedule are held fixed; only contradicted
negative support messages change.

## Broader source-training evidence

Source dependence is firmly established at aggregate scale. In the corrected
three-seed 9×9 final-core matrix, adding a target's source improves PRODIGY NM
accuracy in all 72/72 seeded target-entry transitions, with mean +5.25 points.
Across the specialist matrix, target means, source-model means, and interaction
plus noise account descriptively for 34.98%, 33.13%, and 31.89% of observed
variation. A target-only difficulty explanation is therefore inadequate.

Foreign-source ranking is most strongly associated with feature-distribution
distance (mean within-target Spearman rho −.755 for proxy A), while source node
count and feature effective dimension are strong donor-quality correlates. Raw
homophily difference has no pairwise ranking signal. These nine-graph results
support a two-part account—source coverage/quality plus source-target
compatibility—but remain graph-level correlations among entangled properties.
They do not identify whether the operative episode-level cause is feature-space
coverage, local topology, support conflict, optimization, or some combination.

Existing source-mixture interventions narrow that question. On two source
pairs, source-confined episodes outperform proportional mixed-source episodes;
under extreme source-size imbalance, balanced source exposure rescues the small
source from 0.33 to 0.43 NM accuracy. Those are training interventions, but only
one seed and older 50k/110k endpoints, and they change the distribution of
negatives and per-source exposure together. They support source discrimination
and exposure as real training factors without identifying the episode-level
mechanism behind canonical HK errors.

## Defensible claim boundary

We can explain 6.57% of assigned HK errors with high confidence as metric
artifacts over the full canonical benchmark. We can localize 93.39% of assigned
errors, and 93.29% of errors under corrected multi-positive scoring, to a final
wrong-winner direction carried by the learned positive-support reference. That
is a population-wide computational description, not a causal fraction. For the
first query only, positive-edge ablation supplies direct causal evidence for
the immediate decision. No current result assigns a high-confidence causal
fraction of all HK errors to training conflicts, degree, feature-space support,
or sampled overlap.

The unresolved alternatives are: insufficient source coverage of target
feature/local-structure regions; exclusive supervision on many-to-many
neighborhoods; deterministic member/role policy; high-degree and recurrent-user
sampling; and encoder/readout coadaptation. Existing controls reject simple
versions of several accounts: raw similarity is not the model, negative-message
deletion is not a repair, support replacement is unstable, scalar overlap does
not determine success, and one universal metagraph-only source mechanism is
false.

The matched training pair supplies a controlled negative result. Suppressing
training-view-contradicted negative support messages reduces terminal
multi-positive accuracy from 23.75% to 21.96% and unique-anchor accuracy from
22.51% to 21.19%; 5,303 corrected failures are recovered but 6,404 successes
are broken. Small gains at steps 100 and 300 reverse by step 900. This rejects
message deletion as a sufficient repair, while leaving broader exclusive-target
conflict unresolved because the cross-entropy target itself was unchanged.
The replicated source-manifold test is now a negative result. Across five
balanced, outcome-blind reference banks, absolute target-graph similarity is
only weakly predictive for native HK→HK (node-balanced AUC .562) and inversely
predictive for native Ukraine→Ukraine (.398). Target-minus-other affinity is at
chance or points in the wrong direction in every model-target cell. This weakens
the simple out-of-support account and does not justify a coverage regularizer.
The next cached analysis should compare native/foreign true-versus-rival margins
before and after the metagraph within paired outcome cohorts. That directly
tests whether source training creates episode-relative candidate separation in
the encoder or converts it through the support-derived class reference.

Evidence: [first query and episode](FINDINGS_NM_HK_FIRST_QUERY_EPISODE.md),
[full HK decomposition](FINDINGS_NM_HK_CLASS_REFERENCE_FULL.md),
[native-graph replication](FINDINGS_NM_NATIVE_GRAPH_REFERENCE_REPLICATION.md),
[training-conflict bridge](FINDINGS_NM_HK_TRAINING_CONFLICT_BRIDGE.md),
[matched training intervention](FINDINGS_NM_HK_OVERLAP_TRAINING.md),
[source-manifold diagnostic](FINDINGS_NM_SOURCE_MANIFOLD.md),
[final-core matrix](../../transfer/matrices/cross_model/final_core/FINDINGS.md),
and [nine-graph predictor study](../../graphs/transfer_prediction/similarity_vs_transfer_v2/FINDINGS.md).
