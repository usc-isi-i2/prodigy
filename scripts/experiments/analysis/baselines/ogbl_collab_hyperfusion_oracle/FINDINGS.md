# Compact HyperFusion-protocol test oracle

## Result

The released HyperFusion construction does not beat HyperFusion when applied to the
533,296-parameter AA + structure + joint system. This diagnostic is deliberately
test-informed and is not admissible as held-out benchmark evidence.

| Test-aware rule | Seed 0 | Seed 1 | Seed 2 | Mean ± sample SD |
| --- | ---: | ---: | ---: | ---: |
| Released rule, native scores | 0.63064 | 0.63947 | 0.63412 | 0.63474 ± 0.00445 |
| Released rule, percentile interface | 0.65836 | 0.66263 | 0.66539 | 0.66213 ± 0.00355 |
| Best post-hoc test rank sum | 0.69227 | 0.69207 | 0.69030 | 0.69155 |
| Label-aware union diagnostic | 0.73205 | 0.72933 | 0.73019 | 0.73052 |

HyperFusion's reported target is 0.7129 ± 0.0018. The validation-calibrated AA
component reaches 0.67759 in this replay; the separately test-calibrated AA diagnostic
is 0.68145. Structure experts reach 0.62445--0.63582 and joint experts
0.60295--0.62054 on test.

The released fusion uses cosine similarities of validation-positive,
validation-negative, test-positive, and test-negative score vectors to construct
`H`, then `H @ H.T`, then a weighted sum. With native scores it gives AA zero weight.
With per-model empirical-percentile scores it assigns weights `[6,10,10]`, but the
weaker neural experts reduce performance.

## Decision

Stop global fusion. Even weights optimized directly on test peak near 0.692, well
below 0.7129. The 0.729--0.732 label-aware union is not a realizable scoring rule,
but it establishes that per-pair complementarity is large enough in principle. The
only remaining hypothesis supported by these artifacts is a pair-conditional
selector that identifies expert-only true positives without promoting their hard
negatives. Its first stopping test should be whether a fitted oracle-style gate can
close a substantial fraction of the gap from 0.692 to 0.713; another global ensemble
cannot.

## Cross-fitted pair-gate follow-up

The predeclared five-fold test-informed gate reached its 0.7130 stopping target.
Each unordered edge is assigned deterministically to one fold and is scored by a
model trained on the other four folds. Inputs contain three percentile scores,
absolute score disagreements, and the existing 13 legacy plus 14 symmetric pair
features. Every fold model is counted toward the cap.

| Gate | Seed 0 | Seed 1 | Seed 2 | Mean ± sample SD | Conservative total cap |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic | 0.00000 | 0.00000 | 0.00000 | 0.00000 ± 0.00000 | 533,466 |
| Shallow HistGB | **0.85575** | **0.84543** | **0.84841** | **0.84986 ± 0.00531** | 543,696 |

An exact independent rerun reproduced every row and summary value. Logistic
probabilities saturate/tie at the strict 50th-negative boundary, producing the zero
result; it is not evidence against linear ranking in general.

The tree result beats HyperFusion by 13.70 points under 1M counted scalars, but it is
not leaderboard-valid: each fold uses labels from four-fifths of the official test
panel. It is also more label-informed than the released HyperFusion construction,
which uses test-positive/test-negative score-vector geometry but does not fit a
per-edge classifier. The supported conclusion is that the pair features make the
test labels highly learnable and capacity is not the bottleneck. The next scientific
step is to freeze the identical tree recipe on 2018 labels and apply it once to 2019.

## Evidence and provenance

- Producing revision: `482d5346fe6a293e709c482b9e7da03c6f3172e8`.
- Source checkpoint revision: `be46110e774938c6b567a5340395ea363535ae24`.
- Tucker runtime: `/dataMeR1/phil/gfm/ogbl_collab_hyperfusion_oracle/oracle_v3`.
- Failed setup-only artifacts: `oracle_v1` (validation constant guard) and `oracle_v2`
  (wrong detached Python); neither produced results.
- Protocol SHA-256: `f77c254b671a6a46c516ae69e12a9b5b99d014450b27fc56a97d92da81bbd177`.
- Score archive SHA-256: `8fa02f5e0db8f956ca780c3109d6adfea88501ffe3aa7c5ad669823d2b2d9444`.
- Official test: 46,329 positives and 100,000 shared negatives, validation edges
  appended to the historical graph.
- Gate producing revision: `1bb326c39c7848a48122faf9d0d2d2ca8f31b1c5`.
- Gate runtimes: `/dataMeR1/phil/gfm/ogbl_collab_hyperfusion_oracle/gate_v1` and
  exact replay `gate_v1_repeat`.
