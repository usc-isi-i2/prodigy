# Standard Cora feature-only LP ladder

Status: complete one-seed matched pilot.

## Result

| Encoder | Selected epoch | Validation ROC-AUC | Test ROC-AUC | Test AP |
| --- | ---: | ---: | ---: | ---: |
| Raw 1,433-D features | — | — | 0.7984 | 0.8188 |
| Linear 1,433 -> 128 | 91 | 0.8964 | **0.8821** | **0.8834** |
| Nonlinear 1,433 -> 256 -> 128 | 66 | 0.7974 | 0.8127 | 0.8093 |

The learned linear metric improves test ROC-AUC by 8.38 points over raw cosine.
The nonlinear MLP improves only 1.43 points over raw cosine and trails the linear
arm by 6.95 points. This is evidence for this fixed seed and random-negative panel,
not a claim that nonlinear encoders are generally worse.

## Evidence contract and validation

- Producing revision: `37454797805ef79720851481234bd1dfcf3ca5eb`.
- Source: classic Cora `processed_data.pt`, SHA-256
  `58effe764f4f0a15363d2b168445c2c66dc81220e51621d74156f7139d8ee2ad`;
  verified shape 2,708 x 1,433 with binary features.
- Pair split: seed 0, unique undirected pairs split 85/5/10. Counts are 4,486,
  264, and 528 positives for train, validation, and test, each with an equal-size
  fixed random-nonedge set.
- Pair fingerprint:
  `e34c7c545e5671c3ec0dad7df5caee764599358d684941ed2b52863c8ff54fe9`.
- Runtime gates verified disjoint positive splits, one-to-one positive/negative
  counts, unique nonedges absent from the full graph, expected feature shape and
  binary values, finite scores, and test evaluation only after both selections.
- Both learned arms completed validation-only checkpoint selection. There are no
  missing or rejected cells and no protocol deviations.
- Classification labels and masks, GTE embeddings, graph message passing, and old
  episodic LP evaluation were not used.

The exact protocol, complete validation histories, and final results are under
[`data/`](data/). Checkpoints and pair arrays remain in the Tucker run directory
`/dataMeR1/phil/gfm/cora_standard_mlp_lp/seed0`.
