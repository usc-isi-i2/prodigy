# Canonical NM: query, support, and episode difficulty

8 September 2026. CPU-only analysis of saved canonical-split NM predictions. No model forwards, training, support replacements, or new target episodes. Both models and both targets are included; the tables below retain all four model–target cells.

## Main findings

1. Query difficulty has a persistent component across validation/test, strongest for native Ukraine. Some repeatedly queried nodes are always wrong in the observed test stream; this is not proof they are intrinsically unsolvable.
2. Validation-selected support identities are associated with poorer test predictions, including within-query comparisons. The data do not isolate support harm from anchor identity, the other supports, candidate classes, or sampled contexts.
3. All four queries belonging to one anchor fail together more often than a within-query outcome permutation predicts on both targets. This localizes an association to the episode–anchor class grouping, without identifying its cause.
4. At the full-episode level, additional error clustering is clearer for Ukraine; HK episode dispersion is consistent with the conditional permutation reference. Difficult classes need not make entire 30-class episodes unusually bad.

## Query persistence

Repeated means at least five query occurrences in a split, not necessarily five distinct episodes. Correlations require at least five occurrences in each split. Node identities recur across edge-disjoint validation/test streams; this is not independent-node replication.

| Target | Model | Test nodes queried ≥5 times | Always wrong among these | Val/test eligible nodes | Accuracy Spearman |
|---|---|---:|---:|---:|---:|
| ukr_rus_twitter | ukr | 1523 | 132 (8.7%) | 1100 | 0.659 |
| ukr_rus_twitter | hk | 1523 | 523 (34.3%) | 1100 | 0.324 |
| cp_hk_twitter | ukr | 1526 | 524 (34.3%) | 852 | 0.321 |
| cp_hk_twitter | hk | 1526 | 445 (29.2%) | 852 | 0.321 |

Validation defines hard queries as accuracy ≤20%, easy queries as ≥80%, both with at least five occurrences. Native Ukraine’s hard cohort reaches 16.82% test accuracy (540 nodes; 9,265 occurrences), while its easy cohort reaches 67.64% (125 nodes; 618 occurrences). Native HK’s hard cohort reaches 12.20% (779 nodes; 39,251 occurrences); the easy cohort reaches 33.33%, but only 14 nodes/66 occurrences survive, so its estimate is weak. These contrasts are post-hoc diagnostics, not independently confirmed thresholds.

## Support associations

Each episode–anchor class has three supports and four queries. Assign each support the mean residual correctness of its classes. Expected correctness uses the same query’s OTHER episodes within that split, shrunk with ten pseudo-observations toward its degree × ambiguity × frequency stratum mean. For sparse queries this is largely a coarse baseline. This uses outcomes for diagnosis and is not a deployment feature or causal adjustment.

Among supports appearing in at least ten validation episode–class groups, select the lowest residual quintile, separately per model. Freeze those identities before examining test exposure groups. A test query is exposed when at least one selected identity is in its TRUE class’s three supports; this does not test distracting supports in other classes. The three supports share the same class outcomes and must not be treated as independent observations.

| Target | Model | Selected / eligible supports | Test accuracy: exposed | Test accuracy: other | Same-query difference | Same-query+anchor difference (pairs) |
|---|---|---:|---:|---:|---:|---:|
| ukr_rus_twitter | ukr | 106 / 528 | 23.56% | 48.59% | -8.42 pp | +0.00 pp (2) |
| ukr_rus_twitter | hk | 106 / 528 | 6.08% | 21.14% | -5.81 pp | -23.08 pp (13) |
| cp_hk_twitter | ukr | 89 / 442 | 5.48% | 13.05% | -4.77 pp | -1.03 pp (422) |
| cp_hk_twitter | hk | 89 / 442 | 10.50% | 21.84% | -6.21 pp | -2.32 pp (672) |

Same-query differences equally weight queries observed under both exposure conditions; query+anchor differences equally weight observed pairs under both. These are different restricted populations, not a successive causal decomposition. Native Ukraine has only TWO matched query+anchor pairs: no useful support-specific conclusion. Native HK retains 672 pairs with a −2.32-point difference, versus −6.21 points with query-only matching. Changing anchor/candidate composition remains an important possible explanation of the larger association. No claim of statistically established support harm is made.

## Class failures versus whole episodes

Each target has 512 test episodes, 30 classes per episode, and four queries per class. In 200 deterministic permutations, shuffle correctness among occurrences of the SAME query node. This preserves each node’s total correct/error count, frequency, degree, and the observed query mix of every episode. Singletons cannot move. It breaks association with anchor, supports, sampled contexts and episode; it does not isolate which of those matters. The reference assumes within-query exchangeability for this diagnostic, not randomized experimental assignment.

| Target | Model | Class: all four wrong | Permutation mean | Episode accuracy P10–P90 | Episode SD | Permutation SD mean |
|---|---|---:|---:|---:|---:|---:|
| ukr_rus_twitter | ukr | 26.45% | 21.01% | 35.83–52.50% | 6.51 pp | 5.99 pp |
| ukr_rus_twitter | hk | 58.29% | 54.98% | 13.33–24.08% | 4.22 pp | 4.07 pp |
| cp_hk_twitter | ukr | 72.12% | 67.02% | 7.50–15.83% | 3.38 pp | 3.41 pp |
| cp_hk_twitter | hk | 57.57% | 50.58% | 12.50–23.33% | 3.91 pp | 3.87 pp |

All four observed all-wrong-class fractions exceed the maximum of their 200 permutation references. Whole-episode SD exceeds all 200 references for native Ukraine, 194/200 for HK-on-Ukraine, but only 79/200 for Ukraine-on-HK and 130/200 for native HK. These are exploratory conditional reference comparisons, not independent-seed significance or causal evidence. Models’ episode-accuracy rank correlations are .402 on Ukraine and .427 on HK.

The most useful follow-up is to replay fixed query–anchor cases while changing only their true-class supports or support contexts, retaining the same candidates and realized query inputs. Saved predictions identify candidate groups but cannot perform that intervention. Do not describe difficult queries as harming other predictions or support-associated errors as confirmed harmful supports.

## Evidence and verification

- [Aggregate JSON](data/canonical_split/nm_difficulty_audit.json) retains input SHA-256 hashes, all four model–target cells, full quantiles, exposure cohort sizes, and reference distributions.
- [Read-only analysis](audit_nm_difficulty.py) defaults to the private Tucker export root; override with environment variable `NM_AUDIT_ROOT`. Run in the Tucker `prodigy` environment; output is aggregate JSON on stdout. This does not require loading graphs or GPUs.
- All input rows are unique by split/episode/sample/query. Both splits contain 61,440 rows, 512 episodes, 120 queries per episode, 30 classes and four queries per class. Every class has a consistent support triple.
- All four test scores match the canonical protocol summary. Support cohort sizes and weighted accuracies reconcile with full-test totals; class-failure fractions reconcile to integer counts out of 15,360 classes.
- These are one checkpoint seed per source, reused query identities and exploratory thresholds. No fresh-domain or training-seed replication is asserted.

Read-only input: `/dataMeR1/phil/gfm/error_audit/nm_canonical_split_bios_20260908/{ukr_rus_twitter,cp_hk_twitter}/paired_cluster_queries_private.tsv` (Tucker only). No raw bios or node-level rows were downloaded. Local worktree `/Users/philipp/projects/gfm/prodigy`, branch `main`; no commits or cluster checkout changes.
