# Frozen SAGE layer trace

## Conclusion

The trace does **not** support replacing the encoder with a shallower SAGE. The
third SAGE layer makes endpoint cosine less discriminative, but its representation
raises the unchanged frozen decoder from 66.99%/66.93% Hits@50 at layers 1/2 to
**68.72%** at layer 3. Nearly all of that net benefit is on novel positives.

The most defensible interpretation is that the final layer compresses simple
cosine geometry while creating information the nonlinear product/absolute-
difference decoder can use. There is visible oversmoothing in endpoint cosine,
but it is not currently the dominant performance loss.

## Full-panel results

| Frozen computation | Hits@50 | AUC | Change from full |
| --- | ---: | ---: | ---: |
| Layer 1 output into frozen decoder | 66.9879% | 0.972462 | -1,038 net hits |
| Layer 2 output into frozen decoder | 66.9280% | 0.972929 | -1,074 net hits |
| Layer 3 output into frozen decoder | **68.7155%** | **0.977046** | reference |
| Final layer, learned pair inputs zeroed | 65.9011% | 0.970995 | -1,691 net hits |
| Final layer, structural inputs zeroed | 3.4069% | 0.862579 | -39,240 net hits |

Layers 1 and 2 are off-distribution because the decoder was trained on layer 3.
Likewise, zeroing one decoder block does not estimate a retrained ablation. These
are causal interventions on the frozen computation, not candidate-model results.
The structural-zero collapse shows strong frozen-decoder dependence, but cannot
be read as the performance of an embedding-only architecture.

Removing the learned pair inputs lowers novel-positive recall from 41.31% to
36.03% and zero-AA recall from 20.68% to 15.69%. It loses 1,975 novel hits while
recovering 286, for -1,689 net. Thus the learned representation contributes most
where the model remains weakest. The current failure is not that SAGE contributes
nothing; it contributes materially but still cannot separate enough novel true
pairs from plausible negatives.

## Geometry through the encoder

| Representation scored by endpoint cosine | Hits@50 | AUC | Positive median cosine | Negative median cosine |
| --- | ---: | ---: | ---: | ---: |
| Raw node features | 29.5570% | 0.940920 | 0.9510 | 0.8154 |
| SAGE layer 1 | 25.3345% | 0.940981 | 0.9671 | 0.8018 |
| SAGE layer 2 | 19.6758% | 0.925055 | 0.9869 | 0.8605 |
| SAGE layer 3 | 16.1973% | 0.915132 | 0.9963 | 0.9446 |

By layer 3, both positive and negative endpoint vectors are nearly parallel.
Among the baseline's top 50 negatives, median endpoint cosine progresses from
0.9628 at layer 1 to 0.9842 at layer 2 and 0.9954 at layer 3. The positive/negative
cosine overlap grows monotonically after layer 1. This is descriptive evidence of
oversmoothing in cosine geometry.

However, the frozen decoder uses elementwise products and absolute differences,
not cosine alone. At layer 3 it improves Hits@50 by 1.73--1.79 points relative to
the early-layer substitutions. Layer 3 also lowers the negative cutoff and gives
a net 1,036--1,072 additional novel hits relative to layers 1/2. A shallower model
therefore does not follow from the oversmoothing observation.

## The forty uniformly sampled misses

Only one of the 40 previously sampled baseline misses becomes a hit when layer 1
or layer 2 replaces layer 3. It is validation index 19 (node pair 90145--208320),
which is also the only sampled miss recovered when learned pair inputs are zeroed.
The other 39 remain misses under every frozen intervention. This supports the
full-panel conclusion: there are isolated cases where the final learned channel
hurts, but the larger effect is beneficial.

The sampled cases remain descriptive. They were not used to fit, select, or tune
any diagnostic operation.

## Evidence contract and limitations

The diagnostic loads the admitted fresh-negative joint checkpoint, seed 0 at
update 150, with the exact pre-2018 graph, 2018 panel, standardization and saved
scores. All input and checkpoint hashes match their producing manifests. The local
CPU replay differs from the Tucker GPU logits by at most 1.91e-6 for positives and
2.86e-6 for negatives; the positive hitmask and ordered top-50 negative indices
are exactly unchanged, and official Hits@50 matches. A second local run reproduced
the full result JSON byte for byte.

This is one selected seed and a repeatedly inspected validation panel. It diagnoses
that frozen computation; it does not prove the same layer dynamics across seeds or
years. No labels entered training, probing, calibration, selection, or intervention
choice. No new model was trained. No 2019 data were opened or scored. Prior test
exploration, negative leakage, and test-supervised campaigns remain disclosed.

The Git-bundle transfer to Tucker was rejected by automatic approval review because
the host was treated as unverified for private source history. The safe fallback
used the already retrieved checkpoint and admitted numeric arrays locally. Offline
W&B could not start because the sandbox blocks its local socket; the complete JSON
is the authoritative evidence. These execution constraints did not change model
inputs or the recorded ranking decisions.
