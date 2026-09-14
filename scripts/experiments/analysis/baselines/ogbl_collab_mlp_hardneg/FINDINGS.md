# Feature-only Collab MLP: pair interactions and hard negatives

## Result

Model-mined hard negatives improve the existing nonlinear MLP, while the tested
interaction scorer does not. All values are official test Hits@50, mean and sample
standard deviation across seeds 0--2.

| Scorer | Training negatives | Overall | Seen in training | Novel vs. training |
| --- | --- | ---: | ---: | ---: |
| Cosine | Uniform | 36.95 +/- 1.12 | 67.61 +/- 1.05 | 19.23 +/- 1.20 |
| Cosine | Hard-8 | **38.71 +/- 0.60** | **69.97 +/- 0.82** | **20.65 +/- 0.51** |
| Interaction MLP | Uniform | 36.88 +/- 0.55 | 68.30 +/- 0.85 | 18.73 +/- 0.40 |
| Interaction MLP | Hard-8 | 38.07 +/- 1.06 | **69.97 +/- 1.85** | 19.63 +/- 1.24 |

With the cosine scorer, hard-negative mining gains 1.76 overall Hits@50 points,
2.36 points on recurring pairs, and 1.42 points on novel pairs. The joint
interaction-plus-hard-negative arm improves 1.12 points over the original baseline,
but is 0.64 points below cosine plus hard negatives. The interaction scorer alone is
effectively unchanged overall (-0.07 points) and is worse on novel pairs (-0.51).

## Decision

Keep the shared node MLP and cosine scorer; adopt hard-negative training as the only
supported improvement from this campaign. Do not expand the tested endpoint-product
and absolute-difference interaction MLP without a new reason: it adds parameters but
does not improve the primary result or the novel-link stratum.

The remaining gap to matched GraphSAGE (46.63% overall) is 7.92 points. Hard negatives
repair some ranking mismatch, but they do not eliminate the information advantage of
graph aggregation. The next feature-only iteration, if needed, should test the hard
negative pool size or a ranking loss around the validation top-50 boundary—not a
larger generic pair decoder.

## Evidence contract

- Frozen official temporal split; fingerprint
  `07f7af8e654bda27caad60ed74c479f48780343826543dd613d90cb4e979f9f4`.
- Three seeds per cell. The three cosine-uniform cells are validated reuse from
  `official_v1`; nine new cells were produced at revision `3d06a399`.
- Hard-8 selects the highest-scoring current-model negative from one base uniform
  negative plus seven additional deterministic uniform candidates per positive.
- Positive order, base-negative stream, optimizer, batch size, selection metric, and
  test-closure boundary are matched.
- All 12 semantic cells are present. Selection and checkpoint hashes pass validation.
- The smoke run was validation-only and never scored test.
- Authoritative runtime root:
  `/dataMeR1/phil/gfm/ogbl_collab_mlp_hardneg/hardneg_v1/` on Tucker.
- Machine-readable aggregate: [`data/aggregate.json`](data/aggregate.json).
