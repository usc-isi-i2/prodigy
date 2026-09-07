# GILT scope test: compensation pattern did not replicate

The single predeclared COVID Political to TwiBot-20 test completed at revision
`9c5a456e`, using native-source GILT seed0 checkpoints100/900, 128 original paired
10-shot episodes. Protocol was committed before outcomes at `6bfda494`.

| Encoder | Predictor | Native accuracy | Macro-F1 | AUC | NLL |
|---|---|---:|---:|---:|---:|
| 100 | 100 | 50.4232 | 37.6280 | 50.4937 | 0.697343 |
| 100 | 900 | 50.8138 | 40.4355 | 49.0289 | 0.745499 |
| 900 | 100 | 50.6836 | 38.2620 | 51.2017 | 0.698285 |
| 900 | 900 | 50.7487 | 39.8031 | 49.7884 | 0.767646 |

Percentages except NLL. Later predictor weights improve accuracy slightly with
both encoders (+0.3906/+0.0651 points) but worsen AUC and NLL. Later encoder
weights improve accuracy with the early predictor (+0.2604) and worsen it with
the late predictor (-0.0651); AUC improves with both. Thus the nominated joint
direction fails. E100I900's small accuracy lead is not sufficient evidence of
the predicted mechanism. All native conditions are near chance on this target.

Centered intermediate ridge accuracy is 57.3568/57.1615 for early/late encoders;
macro-F1 is 56.1712/56.0748 and AUC59.2990/59.5052. The same raw-text rule gives
54.5247 accuracy,53.4821 macro-F1,55.8594 AUC. Bypassing native inference helps,
but does not establish the PRODIGY temporal explanation. There is no saved-init
comparison here, so the intermediate-over-text difference is not attributed
solely to pretraining.

All 640 input/output artifact hashes and all512 condition rows passed audit;
every summary mean was recomputed from128 paired rows to absolute tolerance1e-12.
Summary SHA256: `e5d2aa83ea6ec4a89b7931445152c0ce10e4df8ef3ac03ed9c57e89bc22cc2c9`.
Unmodified summary, protocol and completion authority are retained in
`data/gilt_component_crossover_20260907/`. Full tensors remain on Tucker under
`/dataMeR1/phil/gfm/prodigy-gilt-component/log/gilt_component_crossover_20260907/`.
The runner verifies native diagonal parity, encoder/predictor state ownership,
fixed projection, captured support/query features and support prototypes.

Interpretation: compensation is currently a replicated PRODIGY finding, not a
demonstrated architecture-general pattern. This test differs in objective,
training budget, encoder parameterization, normalization and target; it is not
a controlled explanation of which difference causes failure. Its near-chance
native performance also limits what it can establish about competent GILT
transfer generally. Retain this boundary; do not switch to another target or
checkpoint pair to obtain a favorable scope result.
