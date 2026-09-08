# Corrected NM degree audit

8 September 2026. Read-only aggregation of the canonical-split test predictions on Tucker; no model forwards or training. Each target has 61,440 occurrences.

Degree below is full-source incoming plus outgoing edge incidence, not static-train degree or sampled context size. Correctness varies by episode, so node weighting averages each observed node’s correctness before averaging nodes.

| Target | Degree | Occurrences | Nodes | UKR accuracy | HK accuracy | UKR node-weighted | HK node-weighted | Both wrong |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ukr_rus_twitter | 1-9 | 4602 | 4536 | 49.44% | 31.25% | 49.37% | 31.27% | 41.61% |
| ukr_rus_twitter | 10-99 | 9560 | 9322 | 59.53% | 25.84% | 59.50% | 25.86% | 36.64% |
| ukr_rus_twitter | 100-999 | 15056 | 13223 | 54.03% | 20.68% | 53.94% | 20.70% | 41.86% |
| ukr_rus_twitter | 1000-9999 | 14895 | 5983 | 44.74% | 17.86% | 46.11% | 18.68% | 50.57% |
| ukr_rus_twitter | 10000-99999 | 14501 | 888 | 26.50% | 8.96% | 30.74% | 10.96% | 68.82% |
| ukr_rus_twitter | 100000+ | 2826 | 29 | 18.22% | 6.05% | 19.47% | 5.92% | 76.96% |
| cp_hk_twitter | 1-9 | 5275 | 2545 | 40.15% | 44.06% | 39.87% | 43.59% | 43.83% |
| cp_hk_twitter | 10-99 | 6549 | 2294 | 18.58% | 22.42% | 20.54% | 23.93% | 68.67% |
| cp_hk_twitter | 100-999 | 16266 | 1177 | 10.60% | 16.06% | 14.18% | 18.02% | 77.50% |
| cp_hk_twitter | 1000-9999 | 30239 | 205 | 6.52% | 13.60% | 8.64% | 18.83% | 81.81% |
| cp_hk_twitter | 10000-99999 | 3111 | 12 | 4.73% | 14.75% | 27.17% | 42.96% | 81.81% |

## Findings

- Native-source accuracy generally declines toward large degree, but not monotonically: Ukraine peaks at 10–99, and the highest HK bin contains just 12 nodes with a large weighting effect.
- For the native Ukraine model, correct occurrences have median degree 483 versus 3,383 for incorrect occurrences. For native HK, the corresponding medians are 654 versus 1,486. These are occurrence-weighted distributions.
- Excluding within-episode multiple-anchor occurrences does not remove the association. Ukraine native accuracy is 59.53% at degree 10–99 versus 18.48% at 100,000+. HK is 44.07% at 1–9 versus 15.05% at 1,000–9,999.
- Node weighting also retains much of the pattern. However, HK’s 10,000–99,999 group rises from 14.75% occurrence-weighted to 42.96% node-weighted accuracy; do not describe all hubs as uniformly difficult.
- Degree, query repetition, and membership ambiguity remain associated. These strata do not causally isolate degree or explain which source-training property creates an error.

## Evidence

Aggregate counts, degree quantiles conditional on correctness, and single-/multiple-anchor strata: [JSON](data/canonical_split/nm_degree_audit.json). Counts and accuracy totals reconcile with the corrected canonical test exports.

Input: `/dataMeR1/phil/gfm/error_audit/nm_canonical_split_bios_20260908/{ukr_rus_twitter,cp_hk_twitter}/paired_cluster_queries_private.tsv` (Tucker only). Raw rows and IDs were not downloaded. Local branch `main`, worktree `/Users/philipp/projects/gfm/prodigy`.

## Distinct-neighbor verification

Read both full source artifacts on Tucker, counted unique directed and unordered pairs, and verified every test row’s saved incident degree against the graph. Neither artifact contains duplicate directed edge columns or self-loops. Reciprocal edges account for the small difference between incidence and distinct-neighbor counts. Repeated retweet events therefore do not inflate this saved degree measure.

| Graph | Median distinct neighbors | 90th percentile | 99th percentile | Maximum |
|---|---:|---:|---:|---:|
| cp_hk_twitter | 1 | 5 | 71 | 74463 |
| ukr_rus_twitter | 2 | 15 | 178 | 402898 |

These whole-graph distributions differ sharply from the sampled-query distribution. Most graph nodes have few neighbors; NM occurrences disproportionately revisit hubs.

| Distinct neighbors | Ukraine native accuracy | HK native accuracy |
|---|---:|---:|
| 1-9 | 49.49% | 44.05% |
| 10-99 | 59.55% | 22.45% |
| 100-999 | 54.06% | 16.07% |
| 1000-9999 | 44.65% | 13.58% |
| 10000-99999 | 26.48% | 14.75% |
| 100000+ | 18.22% | No sampled queries |

Switching to distinct-neighbor bins leaves the accuracy pattern essentially unchanged. Full distributions and both models’ occurrence-/node-weighted scores: [unique-neighbor audit](data/canonical_split/nm_unique_neighbor_audit.json). These are full-source descriptors, not counts of edges available during message passing.
