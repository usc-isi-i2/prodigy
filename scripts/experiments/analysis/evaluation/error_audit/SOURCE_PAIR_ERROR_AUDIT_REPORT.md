# Source-paired episode error audit

7 September 2026. This report consolidates the Ukraine-versus-Hong Kong
specialist analyses for downstream COVID Political classification and native
30-way neighbor matching (NM).

## Executive findings

1. **The source graph changes which episodes a model can solve, not only its
   aggregate accuracy.** NM shows large native-source specialization on both
   source graphs, and downstream classification shows different structural
   false-positive and false-negative regimes.
2. **A substantial shared hard core remains.** The NM wrong-set Jaccard is
   0.580 on Ukraine and 0.781 on Hong Kong. Source selection alone therefore
   cannot remove most Hong Kong failures.
3. **The specialists are still complementary.** An either-model oracle adds
   3.75 accuracy points on Ukraine and 6.73 points on Hong Kong over the best
   specialist. This makes routing, mixtures, and compact adaptation plausible.
4. **Off-source mistakes are usually more severe.** Even when both models are
   wrong, the native model ranks the true NM anchor substantially higher.
5. **Downstream classification failures expose opposite structural biases.**
   The Hong Kong specialist overcalls connected negatives and misses isolated
   positives; the Ukraine specialist overcalls many isolated negatives.
6. **These results replicate across evaluation samples, not training seeds.**
   NM test/validation results nearly coincide. The classification “original”
   and “fresh” streams use identical checkpoint weights with different episode
   draws. Independent training-seed replication remains outstanding.

## 1. Downstream binary classification

Both frozen specialists were evaluated on exactly paired binary 10-shot COVID
Political episodes. Positive means conservative and negative means
non-conservative. Each of the original and fresh streams contains 3,072 query
occurrences.

![Classification failure distributions](figures/covid_political_fpfn_distributions.png)

### Paired outcomes

| Outcome | Original | Fresh |
|---|---:|---:|
| Both correct | 1,914 | 1,989 |
| Both wrong | 200 | 172 |
| Ukraine only correct | 807 | 762 |
| Hong Kong only correct | 151 | 149 |

The disagreement is asymmetric and stable across episode draws. Most
Ukraine-only wins are negative queries that Hong Kong calls positive: 657 in
the original stream and 652 in the fresh stream. Most Hong-Kong-only wins are
negative queries that Ukraine calls positive: 144 and 138. Positive-class
disagreements are smaller: Hong Kong FN/Ukraine TP occurs 150/110 times, while
Ukraine FN/Hong Kong TP occurs only 7/11 times.

### Structural failure regimes

- **Hong Kong FP / Ukraine TN:** only 10.2%/9.5% are isolated, versus
  45.8%/45.2% of shared true negatives. Mean incoming degree is 3.00/2.80,
  versus 1.49/1.25. Hong Kong therefore tends to overcall connected negatives.
- **Ukraine FP / Hong Kong TN:** 71.5%/78.3% are isolated. Among the small
  connected subset, the positive-neighbor fraction is .446/.378, versus
  .005/.004 for shared true negatives. Ukraine's dominant error regime is
  instead isolated negatives.
- **Hong Kong FN / Ukraine TP:** 52.7%/51.8% are isolated, versus 8.4%/8.1% of
  shared true positives. Mean incoming degree is 1.76/1.75, versus 4.45/4.37.
  Hong Kong disproportionately misses isolated positives.
- **Shared false negatives:** 73.6%/77.5% are isolated, although the cohorts
  are small (53 and 40 occurrences).

Profile availability and profile length do not explain these large cohorts.
All audited rows have row-aligned profile text, and the relevant profile-length
distributions are nearly identical. The evidence instead points toward
different reliance on graph role, query context, or support-derived class
representations.

### Controlled examples

- COVID row 60,598 is a connected non-conservative query with uniformly
  non-conservative sampled neighbors. Ukraine is correct at 84.21%, whereas
  Hong Kong assigns 7.43% to the target. Removing only Hong Kong's query
  background edges makes it correct at 99.41%.
- Row 34,409 is an isolated non-conservative query whose profile explicitly
  supports Biden. Hong Kong is correct at 93.37%; Ukraine assigns 14.91%.
  Zeroing the Hong Kong query bio does not remove its confidence, while changing
  support context flips it. Correctness therefore does not establish direct
  semantic use of the apparently relevant text.
- Row 7,229 is a conservative query with a vague profile but sampled neighbors
  containing conservative cues. Ukraine is correct at 99.65%; replacing only
  its context features drops the target score to 22.20%.

These examples show that both query processing and the support-derived class
representation can mediate the structural associations. Full-graph degree is a
correlate, not yet an established cause.

## Update: first query bio clustering pass

The [query bio clustering report](FINDINGS_QUERY_BIO_CLUSTERS.md) now adds
actual-text interpretation and GTE geometry to the classification audit.
Eight clusters fit without labels/outcomes on original unique queries yield
repeatable error differences in the fresh stream (paired-gap correlation
0.846), although embedding clusters overlap substantially (silhouette 0.031).

The clearest shared-failure concentration is supplied-negative queries in a
cluster mixing pro- and anti-Trump vocabulary: roughly 3% of query occurrences
account for 22% of shared errors in each stream. Inspected bios include both
stance confusion candidates and apparent text/label conflicts, requiring
label-provenance checks before calling these annotation errors. A creative/
hobby cluster retains an additional 4.5/4.6-point Ukraine advantage after a
coarse true-label × isolation adjustment. Ukraine is better in all eight
broad clusters; wholesale routing by bio topic alone is not supported.

![Query bio cluster error rates](figures/query_bio_cluster_errors.png)

This pass covers query bios only. Neighborhood/support GTE distributions,
NM bio clustering, and training-node exposure remain outstanding.

## 2. Native neighbor matching

Both seed-0, step-2,500 specialists were evaluated on exactly paired 30-way,
3-shot NM episodes on `ukr_rus_twitter` and `cp_hk_twitter`. Test and
validation each contain 500 episodes per target. Pairing was verified with the
query node, true anchor, complete episode-anchor map, and true-label support
center-node ids.

![Neighbor-matching paired outcomes](figures/nm_source_pair_outcomes.png)

### Accuracy and reproducibility

| Target / split | Ukraine model | Hong Kong model | Either-model oracle |
|---|---:|---:|---:|
| Ukraine test | 42.68% | 11.36% | 46.43% |
| Ukraine validation | 42.51% | 11.65% | 46.21% |
| Hong Kong test | 13.20% | 20.26% | 26.99% |
| Hong Kong validation | 13.38% | 19.95% | 26.79% |

Ukraine wins all 500 Ukraine test episodes. Hong Kong wins 464 of 500 Hong
Kong test episodes, ties 10, and loses 26. Validation reproduces those counts
almost exactly: 500 Ukraine wins; 464 Hong Kong wins, 9 ties, and 27 losses.

### Failure overlap

| Target | Both correct | Both wrong | Ukraine only | Hong Kong only | Wrong Jaccard |
|---|---:|---:|---:|---:|---:|
| Ukraine test | 7.60% | 53.57% | 35.08% | 3.75% | 0.580 |
| Hong Kong test | 6.47% | 73.01% | 6.73% | 13.79% | 0.781 |

On Ukraine, 93.5% of Ukraine-model errors are shared, but only 60.4% of Hong
Kong-model errors are shared because Ukraine rescues 63,135 query occurrences.
On Hong Kong, shared failures account for 84.1% of Ukraine-model errors and
91.6% of Hong-Kong-model errors. The Hong Kong target therefore has both a
larger common hard core and more oracle headroom.

### Error severity

On shared Ukraine failures, the native Ukraine model gives the true anchor
median rank 5 (mean 6.87), compared with median rank 12 (mean 13.30) under Hong
Kong. On shared Hong Kong failures, the native model's median rank is 7 (mean
8.26), compared with 13 (mean 13.55) under Ukraine.

Thus top-1 accuracy understates useful native-model signal. Rank-aware training,
reranking, or improved support aggregation may recover some shared failures.
The counter-source rescues are also non-negligible—6,757 Ukraine queries and
4,040 Hong Kong queries—providing concrete cases for routing or adaptation.

## 3. Combined interpretation

The consistent explanation is not simply that one checkpoint is globally
better. Source pretraining produces different graph-role and matching
heuristics:

- source matching supplies a large native advantage;
- native representations remain better aligned even on many shared mistakes;
- the specialists preserve complementary decision boundaries; and
- downstream transfer exposes opposing sensitivity to connectedness and
  isolation.

This supports two improvement tracks. A **generalization track** should reduce
source-specific reliance through source-diverse/harder episodes, role-balanced
sampling, context dropout, or invariant structural features. A
**specialization track** should exploit complementarity through routing,
ensembling, rank-aware reranking, or small downstream adapters. The latter may
also support smaller base models or more label-efficient adaptation if the
residual correction boundary is simple.

## 4. What is established versus unresolved

Established:

- exact query/support-center pairing between compared NM models;
- native-source specialization across every or nearly every episode;
- stable test/validation and original/fresh evaluation patterns;
- target-dependent shared-error overlap and oracle headroom; and
- robust structural associations in binary-classification FP/FN cohorts.

Not yet established:

- robustness across independently trained model seeds;
- the fraction of evaluation query, anchor, and support nodes seen during
  pretraining;
- whether node exposure explains source specialization;
- whether degree/connectedness remains predictive after conditioning on the
  exact sampled subgraph; or
- whether query features, graph context, and support aggregation are causal at
  the cohort level.

## 5. Highest-value next analysis

Join the saved episode ids to each source's pretraining exposure ledger and to
target-graph structure. Compare shared failures, native-only rescues, and
counter-source rescues on:

1. query/true-anchor/support-node exposure;
2. degree, directionality, connectedness, and shortest-path status;
3. query-to-true versus query-to-predicted anchor similarity; and
4. within-class support dispersion and query-to-support similarity.

This directly distinguishes coverage or memorization from transferable motif
learning. It also determines whether the next intervention should be broader
pretraining data, structural regularization, mixture/routing machinery, or a
label-efficient adapter.

## Evidence and provenance

Detailed reports:

- [COVID Political classification](FINDINGS_COVID_POLITICAL_SOURCE_PAIR.md)
- [Neighbor matching](FINDINGS_NM_SOURCE_PAIR.md)

Private raw evidence remains under:

- `/dataMeR1/phil/gfm/error_audit/source_pair_fpfn_20260907/`
- `/dataMeR1/phil/gfm/error_audit/nm_source_pair_20260907/`

The classification analysis uses two evaluation episode draws with identical
weights. The NM analysis uses seed-0 specialists and fixed test/validation
episodes. Neither substitutes for independent training-seed replication.
