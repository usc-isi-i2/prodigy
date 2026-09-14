# Warm-positive joint training: failed advancement gate

Training only on previously active endpoints gives **68.5124%** mean selected 2018 Hits@50 versus **68.8908%** for the matched control (-0.3784 percentage points). Every seed regressed. The recipe stops without expansion, refit or 2019 access.

| Seed | Control | Warm-only | Difference (pp) | Recovered / lost |
| --- | ---: | ---: | ---: | ---: |
| 0 | 68.7155% at 150 | 68.6705% at 50 | -0.0449 | 1162 / 1189 |
| 1 | 69.0250% at 100 | 68.2262% at 100 | -0.7989 | 1068 / 1548 |
| 2 | 68.9318% at 100 | 68.6406% at 100 | -0.2913 | 928 / 1103 |

The hypothesis was that excluding historical positives with a cold endpoint would focus training on the evaluation population. All 60084 validation positives have two previously active endpoints, whereas only 47546 of 119622 historical training positives do. Filtering is computed from the pre-2017 graph, and its selected pair set exactly matches the previously documented warm cohort. Validation eligibility and negative membership are unchanged.

## What the error changes show

Novel-pair net hits change by −15, −375 and −132 across seeds; repeated-pair net hits change by −12, −105 and −43. The filtering does not improve novel recall in any seed.

Seed 0 gains 386 nonzero-AA hits but loses 413 zero-AA hits. Seeds 1 and 2 gain 557 and 716 zero-AA hits while losing 1037 and 891 nonzero-AA hits. Thus the training-population change shifts the recovery/retention trade-off without improving overall recall. These are score-based error strata, not causal attribution to individual features. Raw cutoffs across separately trained models are not comparable calibrated quantities.

This rejects the specified warm-only recipe under this budget. It does not show that population mismatch is harmless, that all weighting schemes fail, or that more capacity alone would help. No AA-anchored residual or explicit retention objective was tested here. Earlier failed residual gates and tail objectives remain negative evidence; renaming them is not a new hypothesis.

## Protocol and verification

Six sequential GPU1 cells: control/warm × seeds 0/1/2, 2000 updates each, validation every50, earliest standalone maximum. Architecture, BCE, optimizer, fixed fresh negative panel, graph and features match the existing joint model. Each update samples 2048 positives, 1024 uniform negatives and 1024 mined negatives. Filtering changes positive identities and exposure frequency; this is equal updates, not equal epochs. Hard negatives can differ with model scores. No treatment sample-stream equality is claimed.

All 120 control validation measurements reproduce the archived experiment exactly. All six selected checkpoint replays are exact and match the official OGB evaluator. A separate local audit recomputes six full-panel Hits@50 values using a sorted negative cutoff, verifies earliest checkpoint selection, artifact hashes, revision identity, cohort counts, and all recovery/loss counts. Training assertions verify historical cohort membership and fixed source fingerprints.

Each model retains 496385 neural parameters and 496448 conservative inference scalars. The fixed eligibility rule adds no fitted inference state. No ensemble. Seed variation is optimizer variation on one validation panel, not independent-data confidence. Training/probe metrics contain training positives and are diagnostic only.

Producing revision: f6d5856c7 (full revision in data/protocol.json). Local integrated branch: codex/collab-fresh-ensemble in work/ensemble-repo. Dedicated Tucker branch: codex/collab-warm-joint; worktree /dataMeR1/phil/gfm/prodigy-collab-warm-joint. Full runtime, checkpoints and offline W&B: /dataMeR1/phil/gfm/ogbl_collab_compact_joint/warm_joint_v1. All six jobs completed and tmux exited; no running source was changed. Measured training totals approximately 348 seconds, excluding process startup and audit.

A first local audit invocation used a Python interpreter without NumPy and stopped before computation; rerunning with the repository-prescribed Python 3.11 passed. Plotting required the headless backend; this changed no training or evidence.

## Scope and disclosure

The best previously reported fresh-negative standalone test seed remains 70.1569%, not a new result from this campaign. The 71.29% objective remains unmet. Prior extensive 2019 exploration, negative leakage and test-supervised campaigns remain disclosed. 2018 is repeatedly inspected development data; neither this adaptive experiment nor a fresh freeze erases that history or establishes leaderboard acceptance. Static author features are not historical snapshots. No new test data were opened or scored.
