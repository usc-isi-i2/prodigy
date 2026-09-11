# Political graph bio provenance

## Decision

The Election 2020 and COVID Political graph features are internally aligned with their recorded GTE artifacts, but both embedded an upstream-cleaned `profile` field rather than the original biography text used by the other social graphs. This is a provenance and comparability issue, not a row-alignment defect.

The existing artifacts must remain available as `cleaned-profile-v001` legacy inputs because published experiment runs used them. New work should use raw-profile artifacts when an auditable raw source is available.

## COVID Political

`social_llm_data/covid/full_user_data.csv` contains both `profile` and `raw_profile`. Every one of the 78,672 canonical rows maps uniquely to a full-data row after normalizing the serialized Boolean `verified` field. `scripts/social_llm/prepare_covid_raw_profiles.py` creates a row-identical candidate CSV with only `profile` replaced by `raw_profile`; it rejects ambiguous, missing, empty, or non-round-tripping mappings.

Candidate artifacts live under:

`/dataMeR1/phil/gfm/mixture-scaling/results/political_bio_raw_rebuild_20260911/covid_political/`

The candidate passed full verification: topology, edge attributes, labels, and user IDs are identical to the legacy graph; all 78,672 feature rows equal the new embedding artifact; all stored hashes match the restored raw bios; coordinates are finite; and 77,007 feature rows changed. The report and SHA-256 fingerprints are in `verification.json`. It remains a candidate until promotion is explicitly recorded. Training must not mix the two feature versions without naming the version in the run metadata.

The legacy snapshot is `/dataMeR1/phil/data/covid_political/legacy/cleaned-profile-v001/`. It contains the graph, graph sidecar, embedding, embedding sidecar, source `user_data.csv`, and a verified `SHA256SUMS` manifest.

## Election 2020

No `raw_profile` source was found in `social_llm_data/election2020`, its topology-only `graph.pickle`, or the searched Tucker data/project trees. The existing graph is therefore designated `cleaned-profile-v001`. Do not synthesize punctuation or digits or call a reconstructed string “raw.” A raw-profile replacement requires recovery from the original upstream collection.

Recovery checks also covered the original CARC location `/project2/emiliofe_74/julie/social_llm_data/election2020/`. Its `user_data.csv` and `graph.pickle` are byte-identical to Tucker (SHA-256 `0cb20c0f...304b` and `0fbfd1ec...3273`). The CSV has only `profile,label_conservative`; the graph and edge lists use positional integers and contain no account handle, Twitter user ID, description, or raw-profile attribute. The complete seven-commit history of the public `julie-jiang/retweet-bert` repository contains model code and links to public tweet-ID collections, but no deleted user map or profile data. Public rehydration cannot prove the historical 78,932-account mapping because this derivative discarded account identity. The required recovery input is the Retweet-BERT/Election preprocessing table containing account ID or handle alongside the original profile description and the final positional row assignment.

The immutable reference snapshot is `/dataMeR1/phil/data/election2020/legacy/cleaned-profile-v001/`, with the graph, sidecars, embedding, source CSV, and verified `SHA256SUMS` manifest.

## Evidence

The full audit is in `scripts/experiments/analysis/graphs/features/election_covid_alignment_audit/`. Its principal findings are:

- all graph rows, embedding rows, user IDs, and stored bio hashes align exactly;
- upstream COVID cleaning changes the same-user embedding distribution by mean per-coordinate KS 0.0708;
- political selection and ideology mix are also material: matching Election and COVID Political label proportions reduces their mutual KS from 0.0669 to 0.0403–0.0526;
- re-encoding cleaned COVID profiles reproduces canonical vectors exactly.

The graph catalog carries the short operational warning so future loaders and agents encounter it at the registry entry rather than rediscovering it from geometry plots.
