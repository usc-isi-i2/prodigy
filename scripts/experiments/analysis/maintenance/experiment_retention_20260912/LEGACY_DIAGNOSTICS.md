# Recovered legacy diagnostics

These experiments use the historical rung-9 family and are subject to the Suspended CSV-corruption caveat in the session findings. Retain for provenance, not as a corrected-data benchmark.

The untrained baseline report contains 9 graphs × 5 initializations. Overall mean BCE is 0.838680, range 0.826881–0.859162. These values refer to the report’s historical frozen-check pairs and raw untrained decoder, not the later bias-enabled model.

The dimension-ablation reports pin checkpoint step 859500 and describe disjoint rank/check panels with 1,024 positives and five uniform negatives per positive. Hidden-unit importance was measured by single-unit-removal loss; input ranking used gradient-times-input and random controls. They do not estimate mutual information and do not establish that an independently retrained 64-unit model matches the full model. Both report versions are retained.

Activation reports, paired AUC/BCE and ablation JSONs were untracked on Tucker and are now committed under data/recovered_reports. Their original bytes and origin paths are retained in the compressed snapshot. Reports without a fully revalidated protocol remain historical diagnostics.
