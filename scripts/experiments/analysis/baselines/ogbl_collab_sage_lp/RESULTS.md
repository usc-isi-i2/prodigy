# Matched GraphSAGE-cosine on ogbl-collab

## Result

The two-layer, unique-unweighted-training-graph GraphSAGE encoder improves official
test Hits@50 from `36.95 +/- 1.12%` for the matched nonlinear feature-only MLP to
`46.63 +/- 1.22%` across seeds 0-2, a gain of `9.68` percentage points.

| Model | Hits@10 | Hits@50 | Hits@100 | ROC-AUC | AP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Nonlinear MLP + cosine | 21.24 +/- 1.59 | 36.95 +/- 1.12 | 41.96 +/- 0.86 | 94.32 +/- 0.38 | 92.13 +/- 0.43 |
| GraphSAGE + cosine | **24.97 +/- 1.42** | **46.63 +/- 1.22** | **51.40 +/- 1.19** | **95.32 +/- 0.42** | **93.87 +/- 0.44** |

Values are percent mean and sample standard deviation over three training seeds.
GraphSAGE seed-level test Hits@50 values are 46.44%, 47.93%, and 45.51%.

## Repeat-versus-novel diagnostic

| Model | Seen in training | Novel versus training |
| --- | ---: | ---: |
| Nonlinear MLP + cosine | 67.61 +/- 1.05 | 19.23 +/- 1.20 |
| GraphSAGE + cosine | **88.40 +/- 1.34** | **22.49 +/- 1.17** |

GraphSAGE's overall gain is not purely a new-link gain. It improves novel-pair
Hits@50 by 3.26 points, but improves pairs already observed in training by 20.79
points. The graph supplies useful information for novel collaborations while being
especially effective at recurrence prediction, which is a major component of
`ogbl-collab`.

## Evidence contract

- Producing revision: `8056901e6e8eb4bb5f674dbebfb4bf2d1d36ed60`.
- Dataset split fingerprint: `07f7af8e654bda27caad60ed74c479f48780343826543dd613d90cb4e979f9f4`.
- Unique-unweighted training-graph fingerprint: `c1cbee0a0e0f60f45003551affcce3bc6a916a7f509a33eeb9dc1f612708142f`.
- Official temporal split and OGB evaluator; validation Hits@50 selection; test once
  after selection; no validation-edge reuse.
- All three expected result cells are complete. Both best and last checkpoint hashes
  validated for every seed.
- Runtime artifacts: `/dataMeR1/phil/gfm/ogbl_collab_sage_lp/matched_v1/` on Tucker.
- W&B project: <https://wandb.ai/eibl-usc/ogbl-collab-sage-lp>.

The full-data two-epoch smoke exercised the test path before production. No scientific
setting changed afterward, but the test panel was therefore not literally unseen;
see the setup `DEVIATIONS.md`. This matched arm is also not the official example's
three-layer GraphSAGE plus edge-MLP recipe and should not be labeled as that baseline.

## OGB-style recipe follow-up

The separately named `ogb_style_v1` arm uses the official recipe's major choices:
the supplied weighted training adjacency (including repeated events), three 256-wide
GraphSAGE layers, and a three-layer edge MLP over the elementwise endpoint product.
It keeps validation edges out of message passing and opens test only after selecting
the best validation checkpoint across all 400 epochs.

| Model | Hits@10 | Hits@50 | Hits@100 | ROC-AUC | AP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Matched GraphSAGE + cosine | 24.97 +/- 1.42 | 46.63 +/- 1.22 | 51.40 +/- 1.19 | 95.32 +/- 0.42 | 93.87 +/- 0.44 |
| OGB-style GraphSAGE | 23.92 +/- 1.32 | **48.62 +/- 0.99** | **54.08 +/- 0.53** | **97.10 +/- 0.05** | **95.64 +/- 0.04** |
| Published OGB GraphSAGE | - | 48.10 +/- 0.81 | - | - | - |

The OGB-style arm gains 1.99 Hits@50 points over the matched GraphSAGE and lands
0.52 points above the published mean, well within the variation implied by the two
small seed samples. This reproduces the official baseline closely; it does not
establish superiority. Seed test Hits@50 values are 48.01%, 48.09%, and 49.77%.

The recurrence split remains important: seen-in-training Hits@50 is
`93.76 +/- 0.78%`, while novel-versus-training Hits@50 is `22.54 +/- 1.13%`.
Compared with the matched GraphSAGE, nearly all of the overall improvement comes
from repeat edges; novel performance is essentially unchanged.

Follow-up evidence: revision `8a98ce23c538f78caa935abeeedd6914702faef7`, graph
fingerprint `f951b6169fe0ee95f762680205a5431a70b2d86b92b0226e60a333a693d22485`,
runtime artifacts `/dataMeR1/phil/gfm/ogbl_collab_sage_lp/ogb_style_v1/`.
