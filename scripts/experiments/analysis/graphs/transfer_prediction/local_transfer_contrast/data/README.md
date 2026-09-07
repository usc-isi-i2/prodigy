# Aggregate data

The pairwise tables were refreshed at revision `6ebed263` and the multi-source
tables were produced at that revision from the verified replay at:

`/dataMeR1/phil/gfm/prodigy-local-transfer-contrast/log/local_transfer_contrast/facebook_twibot_vs_election/`

Only aggregate tables are tracked. `occurrences.csv`, `example_candidates.csv`,
the exported tensors, and hydrated example text/identifiers remain in ignored
Tucker experiment output.

The health-predictiveness tables were produced at revision `979ef545` from the
same all-source exports.

The `*_all_seeds.csv`, `*_replication.csv`, and `support_only_*.csv` tables were
produced at revision `9df49785` from 270 fixed-input replay cells (nine sources ×
five targets × three checkpoint seeds × two episode streams). Raw query records,
support embeddings, cached subgraphs, and logits remain in the ignored Tucker
runtime directory `log/local_transfer_contrast/seed_replication/`.

The `schedule_*.csv` tables were produced at revision `c7d307a2` from the six
retained two-source schedule-pilot checkpoints. They are a secondary, one-seed
intervention and are not pooled with the three-seed singleton analysis.
