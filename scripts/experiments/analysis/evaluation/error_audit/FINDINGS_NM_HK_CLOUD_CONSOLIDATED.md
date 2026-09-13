# PRODIGY HK→HK neighbor matching — consolidated research handoff

Consolidated 2026-09-13. This document contains the substantive findings, corrections, experiment methods, limitations, and next-step recommendations from this conversation. It supersedes the initial first-failure report where later adjacency information resolved uncertainty. It is a narrative handoff, not a replacement for the input tensors, per-occurrence predictions, or executable experiment archives.

## 1. Main conclusion

**Hypothesis requiring a new intervention or training test:** the first native HK failure has mixed input evidence. Its pre-metagraph cosine geometry favors invalid support groups, and the native readout does not recover a valid anchor. Explicit sampled cross-subgraph relations allow structural and combined probes trained on other queries to select a valid alternative anchor. This makes preservation/accessibility of relational evidence a plausible mechanism to investigate, but neither encoder information destruction nor metagraph-only failure is established.

**Direct observation from these cached episodes:** structural probes are worse overall than embedding probes. Adding structure to embeddings gives weak, weighting-dependent aggregate changes: geometry gains 12 net valid decisions; the MLP loses 5. Both additive families select a valid anchor for the first case, while their size-only additions fail. This is case-specific evidence, not a demonstrated general model fix.

**Unresolved:** why HK source training produces the observed geometry. This chat used only the supplied HK-labeled encoder cache for the new probes, not a matched comparison of encoders trained on different sources.

## 2. Scope, truth, and evidence labels

Only canonical **Hong Kong→Hong Kong neighbor matching (HK→HK NM)** is considered. NM truth is adjacency to an anchor, not semantic similarity or profile stance. Each episode contains 30 classes, three supports and four queries per class: 210 separately sampled subgraphs, 120 decisions. Every sampled subgraph has a synthetic pooling node.

S encodes each subgraph; U pools it to a data-point embedding; M operates on a bipartite data-point/label metagraph carrying known support assignments. Final prediction uses scaled query–label cosine. Supplied evaluation provenance specifies a seed-0 HK checkpoint at step 2500 and canonical source-disjoint 70/15/15 graph artifact with held-out HK test edges. A config inconsistency is documented below.

Claim labels used here:

- **Direct observation from these cached episodes:** computations performed on uploaded caches/mask, including held-out diagnostic probe predictions.
- **Correlation:** descriptive associations or conditional resampling summaries, not isolated causal effects.
- **Controlled intervention result inherited from prior work:** user-provided earlier interventions, not rerun here.
- **Hypothesis requiring a new intervention or training test:** proposed mechanisms and model implications.

Prior benchmark statistics are explicitly identified as **user-provided prior observations**, not measurements newly established from these 20 episodes.

Two distinct evaluation metrics must remain separate:

1. **Assigned-class accuracy:** match the episode's designated anchor, preserving the canonical reported metric.
2. **Valid-anchor accuracy:** select any anchor adjacent to the query under the supplied undirected static_test mask.

A valid alternative anchor can resolve an adjacency error without resolving the assigned-class error. Supervised probes were trained on valid-anchor targets; their comparison with native is not an architecture-only comparison.

## 3. Inherited background — not rerun here

### Complete 512-episode benchmark

**User-provided prior observations:** 61,440 query occurrences; HK accuracy 17.86%, Ukraine-trained accuracy on HK 11.68%; both fail on 76.01%. HK wins under occurrence weighting, unique-query weighting, and among queries absent from validation. Median HK true rank on failures is 7.

429 queries appearing at least 20 times contribute 70.57% of occurrences. The supplied complete-benchmark ambiguity figure is 23.12%; both models fail on 84.08% of those versus 73.58% of single-anchor occurrences. **Correlation:** repetition and multi-anchor ambiguity predict difficulty but do not fully explain it. These full-benchmark figures are not inferred from this chat's subset. The later supplied subset mask gives a substantially different ambiguity percentage; subset composition and ambiguity-definition alignment should be checked before comparing rates.

### Earlier interventions

**Controlled intervention result inherited from prior work:** on a selected cohort of 100 original failures and 100 correct controls, changing only the known true class's supports while holding queries and competing classes fixed:

- Five random alternative support draws rescue 27/100 failures at least once.
- Random draws retain only 32.6% accuracy on originally correct controls.
- Keeping support identities but resampling their contexts rescues only 5/100 failures.
- Original-input native accuracy is 50.0% versus 19.0% for a simple learned-embedding prototype; the prototype solves only 1/100 original failures.
- On alternative-support draws, native accuracy is 22.4% versus prototype 33.6%. Prototype wins/native fails on 160 draws; native wins/prototype fails on 48.

**Correlation inherited from prior work:** change in encoded margin versus native loss has Spearman −0.522; raw mean-feature margin versus loss has +0.003. Learned compatibility tracks outcomes, but causal mediation is not established.

**Controlled intervention result inherited from prior work:** capped fixed candidate-bank selection using known-true-class learned cosine:

| Selection | Rescued failures | Broke correct controls |
|---|---:|---:|
| Nearest | 25/100 | 54/100 |
| Farthest | 2/100 | 90/100 |
| Diverse | 15/100 | 75/100 |
| Random | 62/500 draws | 330/500 draws |

Nearest rescues only 5/71 previously persistent failures. Original supports remain much better overall. These are diagnostic, oracle-class interventions; no deployable support rule or benchmark improvement was demonstrated. The project should not be reduced to support selection.

**Prior observation/correlation supplied by user:** raw profile-feature similarity often favors wrong-class supports; semantic affinity does not define adjacency and raw feature means poorly track support-resampling outcomes.

## 4. Inputs, schema, and validation

Uploaded original archive: `hk_first20_cloud_bundle.zip`, containing `hk_first20_episodes.pt`, `hk_first20_queries.csv`, private `hk_first20_profiles.csv`, `hk_model_state_dict_2500.ckpt`, `effective_config.json`, `manifest.json`, and README.

**Direct observation from these cached episodes:** all six manifest-listed file sizes and SHA-256 hashes match. Counts agree: 20 episodes; 4,200 subgraphs; 317,852 node occurrences; 313,652 real occurrences; 41,255 distinct real nodes; 334,494 sampled edges; raw features 41,255 × 768.

| Tensor group | Contents / dimensions |
|---|---|
| Node layout | `subgraph_node_ptr` 4,201; occurrence feature indices and IDs 317,852 |
| Sampled edges | `edge_index` 2 × 334,494; `subgraph_edge_ptr` 4,201 |
| Pool edges | Separate `edge_index_supernode` 2 × 4,200 |
| Centers / anchors | 4,200 centers; 20 × 30 anchors; global and HK-local IDs |
| Features | Deduplicated float32 41,255 × 768 |
| Per-episode pre-M embeddings | 210 × 256 |
| Per-episode initial labels | 30 × 256 |
| Per-episode metagraph | 2 × 6,300 edges; 6,300 × 2 attributes; query-edge mask |
| Per-episode outputs | Native, mean-cosine, prototype-cosine: each 120 × 30 |

Class c occupies slots 7c through 7c+6; first three are supports. Real occurrences must be selected by `node_feature_index >= 0`. The pooling feature index is −1, but its HK-local node ID is **−34148423**, so filtering only node ID −1 would incorrectly count the synthetic pool as shared evidence. Reported overlaps exclude it.

Native argmax predictions match all 2,400 CSV rows; recomputed mean support cosine matches the cached head within 1e-6. Trusted local tensor artifacts were loaded with CPU map_location and weights_only=False. No neighborhoods or GNN embeddings were regenerated.

**Provenance caveat:** effective_config.json names a Ukraine run/source/checkpoint path despite the HK-labeled cache/checkpoint. Cached initial labels match the supplied checkpoint's first 30 learned labels exactly; checkpoint cosine scale is 14.77319. This supports cache/checkpoint correspondence but does not independently certify training source. Resolve this upstream before publication. Repository main was inspected as an implementation reference, not pinned as the export's historical commit: https://github.com/usc-isi-i2/prodigy (notably models/general_gnn.py, models/metaGNN.py, models/multilayer_gnn.py).

### Additional valid-anchor export — correction to initial report

Uploaded `hk_nm_first20_valid_anchors.npz` and matching JSON metadata. SHA-256: `91406a6d9fb335b53293359cc214cfdf8fe443432ed6203a9a58c78d58661a8e`.

**Direct observation:** hash, episode order, query/anchor IDs, sample slots, and assigned-class validity align with the original cache. Mask shape is 20 × 120 × 30, with 4,648 positive entries. Its declared convention is undirected canonical HK static_test adjacency; this export was not recomputed from the full graph here.

The mask resolves earlier uncertainty: query **119049** has **three valid anchors, 56905, 29147, 136007**. Prediction **204306 is invalid**. Earlier statements of only two known anchors were lower bounds and are superseded.

## 5. The first failure — complete case summary

**Direct observation:** episode `0:0`, global subgraph/sample slot 3, first query row. All IDs below are HK-local.

- Query **119049**; assigned anchor **56905**, class 0.
- Native prediction **204306**, class 7; true rank **6/30**.
- True logit **10.488414**, predicted logit **11.239079**; margin **−0.750666**.
- Assigned probability **5.981%**, predicted probability **12.670%**.
- Runner-up anchor **328965** has logit 11.239048: winner–runner-up gap **0.00003147**. Exact wrong winner is fragile, though all valid anchors rank lower.

| Native rank | Anchor | Logit |
|---:|---:|---:|
| 1 | 204306 | 11.239079 |
| 2 | 328965 | 11.239048 |
| 3 | 262891 | 11.066248 |
| 4 | 169340 | 10.676739 |
| 5 | 163185 | 10.613610 |
| 6 | 56905 | 10.488414 |
| 7 | 114318 | 10.182880 |
| 8 | 807 | 10.044432 |
| 9 | 75146 | 9.787412 |
| 10 | 148455 | 9.710430 |

### A. Sampled input evidence

**Direct observation:** recorded query graph degree 377; sampled query contains 89 real nodes, 96 stored directed edges, seven zero-feature real nodes. Counts below exclude the synthetic pool; pooling edges are stored separately.

| Role | Slot | Center | Real nodes | Edges | Shared query nodes | Raw cosine | Pre-M cosine |
|---|---:|---:|---:|---:|---:|---:|---:|
| Query | 3 | 119049 | 89 | 96 | — | — | — |
| True support | 0 | 30444 | 84 | 88 | 22 | .5603 | .6652 |
| True support | 1 | 49343 | 81 | 83 | 11 | .5296 | .6239 |
| True support | 2 | 57337 | 74 | 89 | 13 | .6101 | .6386 |
| Predicted support | 49 | 20391 | 73 | 76 | 11 | .5862 | .5484 |
| Predicted support | 50 | 65275 | 71 | 71 | 11 | .5171 | .8424 |
| Predicted support | 51 | 105860 | 74 | 80 | 17 | .6813 | .8493 |

Neither focal anchor 56905 nor 204306 appears in any of these seven subgraphs. Both anchors' cached raw feature vectors are zero and text is missing: substantive query-to-anchor cosine evidence is undefined. Numerical epsilon-stabilized zero must not be interpreted as dissimilarity.

Private profile inspection, without identifying quotations: query concerns China analysis; true supports span HK commentary, advocacy, translation; predicted supports span reporting, diplomacy, regional scholarship. Both groups have topical connections. These descriptions do not define NM truth.

Mean raw center cosine favors predicted supports **.5949 vs .5667**. True class loses to the strongest raw-center competitor by .05292. Full-subgraph raw mean cosine for the true group is **.99156**, yet loses by .002079; context-only mean loses by .002258. Near-parallel mean features provide weak discrimination.

True-support union shares **36** query nodes versus predicted union **29**. Union Jaccard favors true **.13636 vs .11554** and ranks it first among 30, but the next class (anchor 278372) has **.13534**: only **.001025** separation. Query center is absent from all six focal supports but occurs in other classes' support contexts.

**Hypothesis:** identity-aware structural evidence is present but weakly separating. Node-ID intersections are explicit in the audit, not automatically available after independent pooling. We cannot equate a weak diagnostic with intrinsically insufficient model input.

### B. Before the metagraph

**Direct observation:** average encoded query/support cosine is **.64256** for true versus **.74672** for predicted. Two predicted supports are particularly close (.84242 and .84933).

| Head | Winner | Assigned rank | Assigned score | Winning score |
|---|---:|---:|---:|---:|
| Mean cosine | 262891 | 5 | .642564 | .784908 |
| Prototype cosine | 204306 | 5 | .734803 | .860310 |
| Native | 204306 | 6 | 10.488414 | 11.239079 |

**Hypothesis:** the representation expresses compatibility with an invalid support group. This is consistent with imperfect alignment between learned affinity and adjacency, but does not prove semantic shortcut learning or loss of all decodable information. Raw/learned margin magnitudes are not calibrated information-loss measures.

### C. Metagraph/readout

**Direct observation:** native agrees with the prototype's invalid winner; assigned rank changes 5→6. It promotes anchor 328965 from ninth under prototype to nearly tied second. Thus it changes class competition, but does not uniquely introduce the invalid preference.

Metagraph attributes per episode: 90 positive support edges [0,1], 2,610 negative support edges [0,−1], 3,600 query edges [1,0]. Initial label vectors differ by class. Attention weights and post-M vectors are not in the cache; no replay was performed. Attribution to a particular attention message remains unresolved.

### Repetition and controls

**Direct observation:** query 119049 appears four times in the subset and fails all four assigned decisions. In episode 0 it also appears at slot 206 under anchor 136007; both occurrences predict 204306. Their separately sampled pre-M embeddings have cosine **.99560**. All four queries assigned to 56905 and all four assigned to 136007 fail. These are internal observations, not randomized interventions.

Episode 0 has 18/120 assigned-correct and 24/120 valid-correct native decisions. Relevant originally correct controls:

| Query / slot | Degree | Sampled nodes | Subset repetitions | Native confidence | Prototype assigned rank |
|---|---:|---:|---:|---:|---:|
| Failure 119049 / 3 | 377 | 89 | 4 | 12.67% | 5 |
| Correct 69945 / 195 | 290 | 78 | 3 | 10.57% | 18 |
| Correct 36084 / 193 | 1071 | 70 | 3 | 11.85% | 18 |
| Correct 82769 / 67 | 1312 | 89 | 6 | 32.85% | 3 |
| Correct 49343 / 101 | 2620 | 65 | 6 | 14.00% | 14 |

Complete mask gives these controls 1,1,2,6 valid anchors respectively. Earlier observed-membership count of five for 49343 was a lower bound. No exact joint match exists. Native can solve queries with worse cosine ranks than the first failure; low cosine rank alone does not explain failure.

## 6. Patterns across all 20 episodes

**Direct observation:** 2,400 occurrences / **1,034 distinct queries**. Assigned correct **393 (16.375%)**, assigned failures **2,007**, involving **857 distinct queries**. 712 queries always fail in observed occurrences, 145 have mixed outcomes, 177 always succeed. Equal-query assigned accuracy **21.34%**; median native assigned rank on failures **7**.

Using the complete valid mask: **520/2,400 = 21.67%** valid correct, **1,880** valid failures involving **851 distinct queries**. Exactly **127 assigned errors** are alternative-valid predictions. **1,168/2,400 = 48.67%** have multiple valid anchors. Do not generalize this rate to 512 episodes.

Under assigned-class scoring, mean cosine solves 99 and prototype 104 of the 2,007 native failures; either solves **137 occurrences / 123 distinct queries**, both 66. Both fail on **1,870/2,007 (93.17%)** native failures. Across all occurrences, mean/prototype assigned accuracies are 267/2,400 and 277/2,400. These differ from valid-anchor probe results below.

**Correlation:** failed occurrences have greater average recorded degree (3,648 vs 2,485) and subset repetition (7.84 vs 5.20) than correct ones. A strict assigned-class union-Jaccard win occurs in 104/2,007 failures vs 46/393 correct decisions. These are unadjusted occurrence-weighted associations, not isolated causes.

## 7. Probe experiments — methods

Only diagnostic scorers were fitted. No foundation-model training, graph loading, new samples, encoding, or external publication occurred.

**Targets:** 30 binary valid-anchor relations per occurrence, from the supplied held-out static_test mask. Full target adjacency never enters input features. Structural features use only exact cached sample sets/edges, with explicit ID equality/intersection access.

**Fixed evaluation:** episode 0 reserved for case inspection, excluded from fitting/selection. Its query identities excluded from every fit/validation set. Episodes 1–19 partitioned into test groups 1–5,6–10,11–15,16–19. Next cyclic group is validation; remaining groups train. Remove validation queries overlapping test identities; remove training queries overlapping test or validation. Keep all 30 candidates together. Shared support/context identities remain possible: not fully node-disjoint.

Train occurrences by fold: **433,421,429,445**. Final episode-0 fit: episodes 1–15 train and 16–19 validation, with purging, leaving **865 train occurrences / 637 queries**, 376 validation occurrences. Development predictions cover **2,280 occurrences / 1,011 queries**. Episode 0 is already inspected, not a confirmatory holdout. Additive experiments reuse inspected development folds and are exploratory.

**Probe definitions:**

- Jaccard: query node set versus union of a class's support node sets, unfitted.
- Structural logistic: pair intersection/Jaccard, two containment fractions, two center-in-context indicators, immediate sampled-neighbor intersection/Jaccard (direction ignored for this diagnostic). Mean/min/max over supports; union overlap/Jaccard; size descriptors. **34 inputs**.
- Size-only: query nodes/edges; support mean/min/max nodes/edges. **8 inputs**.
- Embedding geometry logistic: query/support cosine, Euclidean distance, vector norms, support/support cosine descriptors. **13 inputs**.
- Pairwise MLP: normalized 256-d embeddings, [q,s,abs(q−s),q*s] → 64 ReLU → 1; average three pair scores. Norm information remains in geometry probe, not normalized MLP.
- Geometry+structure: concatenate to 47 inputs; geometry+sizes control has 21.
- MLP+structure: same embedding network plus zero-initialized linear class-level residual on standardized structural features. Both branches fit jointly; size-only residual control. This is one simple additive interface, not all possible nonlinear interactions.

**Fitting:** positive/negative-balanced binary validity loss over all candidates; training-only standardization. Logistic C∈{.01,1,100}; MLP seed0, AdamW lr=.001, weight_decay=.001, batch32 occurrences; select epoch∈{5,15,40}. Selection uses validation valid-anchor accuracy, ties choose earlier option; no validation refit. Same splits/budgets for compared models. Baseline scores reused in additive follow-up and verified unchanged.

## 8. Unified held-out development results

**Direct observation**, episodes 1–19 only. Equal-query accuracy averages each distinct query's occurrence accuracy, then averages queries. Assigned accuracy remains a separate endpoint.

| model | valid_count | valid_accuracy | equal_query_accuracy | assigned_accuracy |
| --- | --- | --- | --- | --- |
| geometry_plus_structure | 537 | 23.55% | 22.93% | 11.62% |
| geometry_plus_sizes | 520 | 22.81% | 22.62% | 11.93% |
| mlp_plus_structure | 481 | 21.10% | 22.07% | 11.54% |
| mlp_plus_sizes | 467 | 20.48% | 19.86% | 11.05% |
| jaccard | 252 | 11.05% | 7.46% | 6.49% |
| native | 496 | 21.75% | 22.60% | 16.45% |
| mean_cosine | 446 | 19.56% | 20.51% | 11.27% |
| prototype | 472 | 20.70% | 22.07% | 11.67% |
| structural | 371 | 16.27% | 17.88% | 9.61% |
| sizes_only | 84 | 3.68% | 6.70% | 3.33% |
| embedding_geometry | 525 | 23.03% | 23.12% | 11.97% |
| embedding_mlp | 486 | 21.32% | 20.74% | 11.49% |

### Paired wins/losses

**Direct observation:** structural-only versus embedding MLP: 210 structure-only wins, 325 MLP-only wins, 161 both valid, 1,584 neither. Structure beats size-only on 348 occurrences and loses on 61. Structure is valid while both learned embedding probes fail on **170 occurrences / 139 distinct queries**. This is complementarity, not general superiority.

Structure is below MLP in every outer fold (18.17 vs 21.83%; 15.83 vs 21.17%; 15.83 vs 22.67%; 15.00 vs 19.17%). MLP versus native has 243 rescues and 253 breaks.

Additive comparisons:

| Addition vs baseline | Rescues | Breaks | Net occurrences | Equal-query change |
|---|---:|---:|---:|---:|
| Geometry+structure vs geometry | 173 | 161 | +12 | −0.18 pp |
| Geometry+structure vs geometry+sizes | 171 | 154 | +17 | +0.31 pp |
| MLP+structure vs MLP | 195 | 200 | −5 | +1.33 pp |
| MLP+structure vs MLP+sizes | 216 | 202 | +14 | +2.21 pp |

Geometry's structure-addition fold changes: **−0.33,+2.67,0.00,−0.42 pp**; aggregate gain comes from one fold. MLP's changes: tied, worse, worse, better. No robust general incremental advantage over embedding-only scorers is demonstrated.

**Correlation / conditional uncertainty:** query-cluster bootstrap equal-query intervals for geometry+structure minus geometry are **[−2.44,+2.12] pp**; for MLP+structure minus MLP **[−0.78,+3.52] pp**. Both include zero. These condition on fixed fitted predictions and omit training uncertainty and residual shared-episode/support dependence. They are exploratory resampling summaries, not confirmatory significance tests.

In the initial pilot, structure-minus-MLP equal-query difference was −2.86 pp with interval [−5.41,−0.25]; structure-minus-size-only +11.18 [8.76,13.65]. First-index versus uniformly random exact tie handling barely changes initial structural/Jaccard valid accuracy: 16.272% and 11.054% expected, respectively. No broad tie-sensitivity conclusion was newly tested for additive models.

Higher valid accuracy than native is not a canonical benchmark fix: valid-anchor-supervised probes have lower assigned accuracy and were not evaluated on the complete benchmark.

## 9. First failure after probes

**Direct observation:** all models below use the separate episode-0 fit where applicable.

| model | predicted_anchor | valid | assigned | best_valid_rank | valid_margin |
| --- | --- | --- | --- | --- | --- |
| geometry_plus_structure | 29147 | 1 | 0 | 1 | 0.143393 |
| geometry_plus_sizes | 262891 | 0 | 0 | 9 | -0.943127 |
| mlp_plus_structure | 29147 | 1 | 0 | 1 | 0.307587 |
| mlp_plus_sizes | 75146 | 0 | 0 | 2 | -0.009091 |
| jaccard | 56905 | 1 | 1 | 1 | 0.001025 |
| native | 204306 | 0 | 0 | 6 | -0.750666 |
| mean_cosine | 262891 | 0 | 0 | 5 | -0.142344 |
| prototype | 204306 | 0 | 0 | 5 | -0.125507 |
| structural | 29147 | 1 | 0 | 1 | 0.692728 |
| sizes_only | 2686 | 0 | 0 | 14 | -0.667527 |
| embedding_geometry | 262891 | 0 | 0 | 8 | -0.909133 |
| embedding_mlp | 75146 | 0 | 0 | 2 | -0.002799 |

Both combined families select **29147**, a valid alternative. Neither size addition does. Structural-only also selects 29147; simple Jaccard selects assigned 56905. Thus learned-structural rescues are not assigned-class repairs.

The valid 29147 support group has 35 query-union shared nodes versus native wrong group's 29; one support context contains the query center, and one pair has one immediate sampled-neighbor intersection. These are explicit relational clues. Structural scores mix correlated size/overlap terms with cancellation; no single-feature causal attribution is justified. The normalized embedding MLP's original miss is close: best valid rank2, margin −.002799. Combined margins have different scales and should not be compared as calibrated confidence.

Correct controls 69945 and 36084 are missed by structural-only and both initial learned embedding probes. All those probes select valid anchors for controls 82769 and 49343. The initial-case rescue therefore coexists with losses on relevant controls.

**A — Direct observation:** sampled inputs contain a relation signal that an identity-aware rule fitted elsewhere can exploit to pick a valid anchor. This does not certify unique assigned-class evidence or its accessibility through the existing architecture.

**B — Hypothesis:** some cross-subgraph evidence may be hard to access after independent pooling. Missing under two limited embedding probes is not proof that embeddings lack it; the MLP is close and embedding probes perform better overall.

**C — Direct observation / hypothesis boundary:** native and embedding probes have two-way disagreements. No tested initial embedding probe solves this first case. Metagraph-only blame is unsupported; message-level attribution would require replay.

## 10. What remains worth testing

**Hypotheses requiring a new intervention or training test:**

1. Stop tuning on these same 20 episodes. Use additional existing caches and untouched evaluation episodes to assess whether complementary structural signals generalize with adequate fitting data.
2. To explain source dependence, encode the **same exact inputs** with HK- and Ukraine-trained checkpoints and run a fixed probe protocol. Compare where source differences appear before M and in native outputs; keep input evidence and evaluation targets fixed. This comparison was proposed, not performed.
3. If localizing the readout further, replay the exact supplied checkpoint from cached pre-M inputs, first verifying native logits, then inspect post-M representations and controlled perturbations. Replay was not performed here.
4. Full message-passing adjacency, separately from held-out truth, would allow comparison of structural evidence before versus after sampling. The valid mask alone cannot quantify evidence omitted by sampling. This graph-level audit was proposed, not performed.
5. Any relation-preserving model change must use information legitimately available under the canonical message-passing split. Full held-out target edges cannot become input features. Explicit sample-ID overlap is a diagnostic access difference that must be acknowledged when designing an interface.

Do not claim that support replacement improves the benchmark, learned cosine causally determines success, the metagraph alone causes ordinary failures, profile semantics define truth, this subset represents the full benchmark, or that any model fix has been demonstrated.

## 11. Artifacts and continuation status

No files from this chat were committed or pushed to Git. The analysis workspace was not a Git repository; the PRODIGY repository was not modified.

Completed artifacts:

- `hk_first_failure_report.md`: initial case report, superseded by this consolidation for complete-anchor validity and later experiments.
- `hk_probe_results.zip`: initial probe report, scripts, feature cache, scores, predictions, split/selection logs, and summaries. Initial fitted probe weights were not retained in that archive; deterministic fitting scripts/settings were retained.
- `hk_additive_probe_results.zip`: additive report, scripts, fitted weights/scalers, scores, predictions, selection logs, environment information, and protocol.
- This document: unified findings and handoff. It does not embed private profiles, original tensors, all score matrices, or executable source code.

To reproduce or resume computationally, retain the original bundle, valid-anchor NPZ/JSON, and both results archives. To resume discussion, this Markdown is sufficient context for the completed findings and unresolved questions.

Input/artifact hashes and exact paths are recorded in manifests and archive `input_hashes.json` files. Private profile text must remain private and should not be committed publicly. Historical repository commit and checkpoint training-source provenance remain unresolved as noted above.
