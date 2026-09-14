# Validation-frozen compact expert fusion on ogbl-collab

## Decision

Stop this fusion family. The predeclared advancement rule failed. The rule selected
on 2017 did improve over AA-DC on 2018, but it did not improve by the required one
percentage point and it did not beat both single-expert controls in every seed.
No test labels or test scores were used.

## Frozen protocol

- Source checkpoints: compact-joint revision `be46110e774938c6b567a5340395ea363535ae24`.
- Fusion implementation revision: `623b1adb7a09258567db0fe667e5e11ef79df842`.
- Selection year: 2017. Assessment year: 2018 official validation split.
- All score scales and one common alpha tuple were selected on 2017 and then frozen.
- Learned inference scalars for AA + structure + joint: 533,296 (<1M).
- Runtime evidence: `/dataMeR1/phil/gfm/ogbl_collab_compact_fusion/fusion_v2` on Tucker.
- This remains post-hoc development evidence because 2018 had been inspected by
  earlier campaigns.

## Selection result (2017)

| Rule | Frozen weights (structure, joint) | Mean Hits@50 |
| --- | ---: | ---: |
| AA + structure | (1.0, 0.0) | 0.77359 |
| AA + joint | (0.0, 1.0) | 0.76711 |
| AA + both | (0.5, 1.0) | **0.78703** |

## Assessment result (2018)

| Rule | Seed 0 | Seed 1 | Seed 2 | Mean ± sample SD |
| --- | ---: | ---: | ---: | ---: |
| AA + structure | 0.67867 | 0.67988 | 0.66996 | 0.67617 ± 0.00541 |
| AA + joint | 0.67554 | 0.68978 | 0.68198 | **0.68243 ± 0.00713** |
| AA + both | 0.67409 | 0.68386 | 0.67539 | 0.67778 ± 0.00531 |

The frozen-2015-calibrated AA baseline is 0.66840. The separately reported
official-2018-calibrated AA baseline is 0.67356. Thus the selected two-expert rule
adds 0.94 points over its matched frozen baseline, but only 0.42 points over the
official AA number. It also underperforms the joint-only control by 0.47 points.

The two-expert rule recovered 2,142 / 2,421 / 2,335 AA misses, but lost 1,800 /
1,492 / 1,915 AA hits. The net gains were 342 / 929 / 420 positives.

## Interpretation

The apparent complementarity on 2017 does not identify a stable fusion rule. The
ranking reverses on 2018: both experts are best in selection, whereas joint-only is
best in assessment. This is evidence against spending more compute on fixed global
max-fusion weights. The bottleneck is temporal calibration/gating, not insufficient
capacity: the compact experts already recover many AA false negatives but promote
too many negatives or displace AA true positives when their global scale transfers.

This experiment does not support a claim that the compact system beats HyperFusion.
The next credible route must learn a pair-conditional abstention/gate under a
strictly forward-chained protocol, or establish a clean comparison showing that
HyperFusion's published test-informed fusion is not a comparable evaluation target.

## Evidence receipt

- `results.json` SHA-256: `01f26c68bc000eb80ef23f8c3798d43f251ed9af1a598467cfe9681da99263e5`
- `selection_frozen.json` SHA-256: `e517e94d6012ee169a42ec573d0e4c4628f28d4ab23d38a42ad487ee52084fea`
- `protocol.json` SHA-256: `60fc9352a8fe088e7c30bc245dee3474a78fc36a50f29720c7747defed45e964`
- `audit.json` SHA-256: `71bbc430f81174f419dc2295c6107b2153727d6f1ce2b1ca99e7a99de24cb0d2`
- Built-in audit verified the frozen receipt chain, nine assessment cells, parameter
  cap, and `test_scored: false`.
