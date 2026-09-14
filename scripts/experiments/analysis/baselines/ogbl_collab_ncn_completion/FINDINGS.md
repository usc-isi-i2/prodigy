# Compact NCN and one-step completion: completed comparison

All nine predeclared cells completed. No new arm passed the advancement gate; stop without expansion, refit or test scoring.

| Model | Mean selected 2018 Hits@50 | Difference from control |
|---|---:|---:|
| Joint control | 68.8908% | +0.0000 pp |
| Observed neighbors | 63.9055% | -4.9852 pp |
| With completion | 61.7247% | -7.1661 pp |

| Seed | Control | Observed | Completion |
|---|---:|---:|---:|
| 0 | 68.7155% at update 150 | 63.5993% at update 200 | 62.0648% at update 250 |
| 1 | 69.0250% at update 100 | 63.9288% at update 450 | 60.8831% at update 650 |
| 2 | 68.9318% at update 100 | 64.1885% at update 400 | 62.2262% at update 650 |

## Where the predictions change

Each stratum below reports its own positive recall at that model’s recomputed full negative cutoff, averaged over all three seeds. Strata overlap and must not be added.

| Positive stratum | Control | Observed | Completion |
|---|---:|---:|---:|
| novel | 41.6302% | 34.5808% | 32.8346% |
| repeat | 99.9774% | 97.3461% | 94.6697% |
| zero_aa | 19.5867% | 0.6003% | 11.5508% |
| nonzero_aa | 95.5784% | 98.1719% | 88.8832% |

| Comparison | Seed | Recovered positives | Lost positives | Net |
|---|---:|---:|---:|---:|
| control → observed | 0 | 1590 | 4664 | -3074 |
| control → completion | 0 | 1212 | 5208 | -3996 |
| observed → completion | 0 | 2563 | 3485 | -922 |
| control → observed | 1 | 1282 | 4344 | -3062 |
| control → completion | 1 | 993 | 5885 | -4892 |
| observed → completion | 1 | 2408 | 4238 | -1830 |
| control → observed | 2 | 1274 | 4124 | -2850 |
| control → completion | 2 | 897 | 4926 | -4029 |
| observed → completion | 2 | 2474 | 3653 | -1179 |

## Interpretation and important limitations

Across seeds, observed pooling raises nonzero-AA recall from 95.5784% to 98.1719%, but zero-AA recall falls from 19.5867% to 0.6003%. Completion recovers some of the latter (11.5508%) while lowering nonzero-AA recall to 88.8832%. Completion therefore loses overall even against observed pooling, and neither new arm improves novel-collaboration recall over control. These are net ranking outcomes, not causal attribution to individual features.

The observed-neighbor decoder and completion decoder have the same neural architecture and initialization; their difference tests the declared completion computation. The comparison against the existing joint control is broader: the new decoders replace its 14 handcrafted inputs with learned neighbor pooling. They do not simply add information to an otherwise identical predictor. The results cannot isolate the effect of removing those features from the effect of adding the neighborhood representation.

This is an NCNC-inspired compact adaptation, not a reproduction of the published NCNC model. The encoder is retained SAGE, the head is a compact MLP, common neighbors are capped at 32 and one-sided candidates at 8 per endpoint with label-independent sampling and population-mass expansion. Completion uses the shared observed decoder at one recursion level with detached sigmoid(logit+log(.1)) weights. Training weights refresh every 10 updates; validation weights are fresh. The fixed prior, sampling approximation, architecture and optimization may limit performance. Do not generalize a negative result here to every NCNC architecture or input combination.

The 40 inspected misses were not a fitted training subset. All 119,622 historical positives and the same fresh 100,000-negative panel were used. No warm-only filtering, renewed negatives or new tail objective was introduced. Supplied static author features are not historical snapshots.

## Protocol and evidence integrity

All nine cells use 2000 updates and validation every 50, earliest standalone maximum per cell, balanced BCE, Adam .001, weight decay .0001, gradient clip 5, 2048 positives/1024 uniform negatives/1024 mined negatives and top 2048 mining refreshed every 10. The advancement rule requires every seed to improve over control and mean gain>=.005. Each arm is judged against control; if both pass, higher mean wins, observed breaks exact ties.

All 120 control validation measurements exactly reproduce the archived control. All nine selected checkpoint replays are exact and agree with the official OGB evaluator. The independent audit checks artifact hashes, source/cache manifest identity, earliest selections, full-panel sorted-cutoff metrics, all error counts, equal sampling-slot streams and paired encoder initialization. Actual mined-negative identities may differ with model scores.

Four synthetic contract tests passed: graph/common/candidate membership and exclusions; symmetry, padding, zero-completion equality and finite gradients; sampling caps with mass expansion; exact encoder initialization and model count. Separate cache audits checked 138 training and 139 validation target rows, including capped cases, against graph neighborhoods and checked their represented child links. Full input pair arrays match the admitted panels. Three ten-update training-only smoke cells and a separate concurrent profile passed before production. No validation score was used to adjust the model or schedule after launch.

The control has 496,385 neural parameters (496,448 conservative inference scalars); both new models have 525,633 neural parameters (525,697 conservative scalars, including a deliberately conservative allowance for old auxiliary state and the fixed prior). Only one selected model is used for inference. No learned node-ID table or external pretrained model is hidden in graph caches. The parameter limit is not a runtime limit: completion additionally scores 1,517,908 cached child rows (including one padding self-pair) for the 160,084-pair validation panel; observed pooling does not.

| Arm | Mean measured training/evaluation seconds per seed | Peak allocated GPU bytes |
|---|---:|---:|
| Joint control | 56.99 | 2513655808 |
| Observed neighbors | 185.90 | 3665525760 |
| With completion | 281.48 | 3672656896 |

Producing revision: 6e0963a2d54d1a723467558e2d977c63fbd2af4c. Dedicated Tucker branch codex/collab-ncn-completion, worktree /dataMeR1/phil/gfm/prodigy-collab-ncn-completion. Runtime and full checkpoints: /dataMeR1/phil/gfm/ogbl_collab_compact_joint/ncn_completion_v1. Production ran sequentially on GPU 1; other owned GPUs had active jobs. Offline W&B paths and exact source hashes are retained in each result/config. Local integrated source and evidence are on codex/collab-fresh-ensemble in work/ensemble-repo.

An initial control-archive transfer placed the tar exclusion option after positional arguments. Tar reported the error and included extra W&B files; the expected control artifacts were still present and verified. Subsequent transfers corrected the option order. No training or evidence values changed. A read-only progress command also encountered unavailable rg on Tucker and used grep subsequently.

## Test and benchmark status

No 2019 data were opened or scored in this campaign. The prior fresh-negative standalone best test seed remains 70.1569%; the 71.29% objective remains unmet. Previous extensive test exploration, near-complete negative leakage and test-supervised campaigns remain disclosed. 2018 is repeatedly inspected adaptive development data. No leaderboard acceptance claim or submission follows from this experiment. Seed dispersion reflects optimization on shared data, not independent-data confidence.

Method reference: [NCNC paper](https://proceedings.iclr.cc/paper_files/paper/2024/hash/3efb4bdc6bfe13e1ff95b4407c37961d-Abstract-Conference.html).
