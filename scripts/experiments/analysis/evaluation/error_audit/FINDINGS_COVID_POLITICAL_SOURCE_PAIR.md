# Source-paired COVID Political error audit

7 September 2026. **Exploratory fixed-checkpoint diagnosis, not an independent
training replication or a causal pretraining-data intervention.**

## Question and protocol

What kinds of downstream classification queries distinguish a PRODIGY model
pretrained by neighbor matching on Ukraine from one pretrained on Hong Kong?
Both frozen singleton specialists are evaluated on exactly the same binary
10-shot COVID Political episodes. The positive dataset label is conservative;
the negative label is non-conservative.

The audit contains 3,072 query occurrences in each of two evaluation streams:
the established original stream and a fresh stream generated with a different
episode-seed offset. The checkpoint weights are identical across streams. These
are two episode draws, not two training seeds. Across the 6,144 occurrences,
5,113 query node identities are represented and 484 identities occur in both
streams.

Each private row retains stream, episode, query node id, target, both models'
predictions and scores, TP/TN/FP/FN outcome, and paired correctness cohort. It
is joined to the target graph for full-graph incoming/outgoing degree, weighted
retweet/mention activity, neighbor-label composition, and the row-aligned
profile text used to construct the graph feature. Raw identifiers and profiles
remain on Tucker and are not committed.

## Main result: opposite structural failure regimes

The disagreement is highly asymmetric and replicates across episode streams.

![Classification failure distributions](figures/covid_political_fpfn_distributions.png)

| Paired outcome | Original | Fresh |
|---|---:|---:|
| Both correct | 1,914 | 1,989 |
| Both wrong | 200 | 172 |
| Ukraine only correct | 807 | 762 |
| Hong Kong only correct | 151 | 149 |

Most Ukraine-only wins are negative queries that Hong Kong calls positive:
657/652 Hong Kong FP–Ukraine TN occurrences. Conversely, most Hong Kong-only
wins are negative queries that Ukraine calls positive: 144/138 Ukraine
FP–Hong Kong TN occurrences. Positive-class disagreements are smaller:
Hong Kong FN–Ukraine TP is 150/110 and Ukraine FN–Hong Kong TP is 7/11.

### Hong Kong false positives are connected negatives

Among Hong Kong FP–Ukraine TN queries, only 10.2%/9.5% are fully isolated,
versus 45.8%/45.2% among shared true negatives. Mean full-graph incoming degree
is 3.00/2.80 rather than 1.49/1.25. Their incoming-degree distributions are:

| Incoming degree | Original | Fresh |
|---|---:|---:|
| 0 | 11.9% | 11.7% |
| 1 | 40.6% | 42.2% |
| 2–3 | 26.9% | 26.4% |
| 4–7 | 12.3% | 12.1% |
| 8+ | 8.2% | 7.7% |

### Ukraine false positives are mostly isolated negatives

Ukraine FP–Hong Kong TN queries show nearly the reverse pattern: 71.5%/78.3%
are fully isolated. Their incoming-degree distributions are:

| Incoming degree | Original | Fresh |
|---|---:|---:|
| 0 | 75.7% | 81.9% |
| 1 | 13.2% | 11.6% |
| 2–3 | 9.0% | 5.1% |
| 4+ | 2.1% | 1.4% |

Among the minority with incoming neighbors, the mean fraction of positive
neighbors is .446/.378, compared with .005/.004 for shared true negatives.
Only 35/25 Ukraine false positives have incoming neighbors, so this is a
candidate mechanism rather than a stable population estimate.

### Hong Kong false negatives are isolated positives

Among Hong Kong FN–Ukraine TP queries, 52.7%/51.8% are fully isolated, versus
8.4%/8.1% among shared true positives. Their mean incoming degree is 1.76/1.75,
versus 4.45/4.37 for shared true positives. Shared false negatives are even
more concentrated in isolation: 73.6% of 53 original and 77.5% of 40 fresh
occurrences.

Profile availability and length do not explain the large cohorts. All audited
query rows have row-aligned profile text. Hong Kong FP and shared-TN profile
length distributions are nearly identical, as are Hong Kong FN and shared-TP
distributions.

## Concrete examples and computational caution

The earlier identity-verified qualitative audit contains cases consistent with
these regimes:

- COVID row 60,598 is annotated non-conservative and explicitly describes being
  a Texas liberal; all 20 sampled neighbors share the non-conservative label.
  Ukraine is correct at 84.21%, while Hong Kong assigns 7.43% to the annotated
  class. Removing only Hong Kong's query background edges makes it correct at
  99.41%, retaining the same query text, sampled nodes, and supports.
- Row 34,409 is an isolated non-conservative query whose bio explicitly says
  "Supporting Biden." Hong Kong is correct at 93.37% for the annotated class;
  Ukraine assigns 14.91%. Yet zeroing Hong Kong's query bio makes it still more
  confident, while changing only support context flips it. Correctness does not
  prove direct semantic use of the profile.
- Row 7,229 is a conservative query with a vague profile and sampled neighbors
  containing explicit conservative cues. Ukraine is correct at 99.65%; replacing
  only its context features drops the annotated-class score to 22.20%. Context
  can therefore supply real signal even though it harms other cases.

These selected examples establish that the observed structural associations can
be mediated by query processing or by the support-derived class representation.
They do not establish that full-graph degree itself caused a decision.

## Interpretation and actionable next test

The working hypothesis is that source pretraining produces different reliance
on target graph role. Hong Kong overcalls connected/incoming negative accounts
and misses isolated positives; Ukraine handles those cohorts better but is more
vulnerable on isolated negatives and a small set of negatives embedded near
positive-labeled neighbors.

This hypothesis is actionable through role-balanced episode sampling, directed
role features, context dropout, degree-conditioned calibration, or a lightweight
adaptation head. It is not hardened enough to choose among them because the
current structural columns describe the full graph rather than each exact
sampled query/support subgraph. The discriminating follow-up is to export exact
sampled context and support identities for these paired cohorts, then test
whether isolation/incoming-role effects survive conditioning on the realized
episode input and across independently trained checkpoints.

## Private evidence

- Paired rows:
  `/dataMeR1/phil/gfm/error_audit/source_pair_fpfn_20260907/covid_political_ukr_vs_cp_hk_query_occurrences.tsv`
- Enriched rows with profiles:
  `/dataMeR1/phil/gfm/error_audit/source_pair_fpfn_20260907/covid_political_ukr_vs_cp_hk_enriched_private.tsv`
- Numeric cohort summary:
  `/dataMeR1/phil/gfm/error_audit/source_pair_fpfn_20260907/numeric_cohort_summary.tsv`
- Selected high-confidence cases:
  `/dataMeR1/phil/gfm/error_audit/source_pair_fpfn_20260907/representative_high_confidence_examples_private.tsv`

The base paired TSV SHA-256 is
`994f709188838cce91ac204940ee526bbd18e322038628a73bf086ac729f6495`.
The full-graph statistics use
`/dataMeR1/phil/data/covid_political/graphs/retweet_graph.pt`; profiles are joined
by row index to `/dataMeR1/phil/data/social_llm_data/covid/user_data.csv`.
