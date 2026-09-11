# Political graph bio provenance

## Decision

The Election 2020 and COVID Political graph features are internally aligned with their recorded GTE artifacts, but both embedded an upstream-cleaned `profile` field rather than the original biography text used by the other social graphs. This is a provenance and comparability issue, not a row-alignment defect.

The existing artifacts remain available as `cleaned-profile-v001` legacy inputs because published experiment runs used them. New work uses raw-profile artifacts when an auditable raw source is available.

## COVID Political

`social_llm_data/covid/full_user_data.csv` contains both `profile` and `raw_profile`. Every one of the 78,672 canonical rows maps uniquely to a full-data row after normalizing the serialized Boolean `verified` field. `scripts/social_llm/prepare_covid_raw_profiles.py` creates a row-identical candidate CSV with only `profile` replaced by `raw_profile`; it rejects ambiguous, missing, empty, or non-round-tripping mappings.

The reproducible build receipt and pre-promotion candidate remain under:

`/dataMeR1/phil/gfm/mixture-scaling/results/political_bio_raw_rebuild_20260911/covid_political/`

The candidate passed full verification: topology, edge attributes, labels, and user IDs are identical to the legacy graph; all 78,672 feature rows equal the new embedding artifact; all stored hashes match the restored raw bios; coordinates are finite; and 77,007 feature rows changed. The report and SHA-256 fingerprints are in `verification.json`.

`raw-profile-v001` was promoted atomically to the canonical Tucker paths on 2026-09-11. The canonical graph is `/dataMeR1/phil/data/covid_political/graphs/retweet_graph.pt`, its embedding is `/dataMeR1/phil/data/covid_political/embeddings/user_bio_embeddings_gte_multilingual_base.pt`, and its row-aligned source is `/dataMeR1/phil/data/social_llm_data/covid/user_data.csv`. Post-promotion verification independently loaded these canonical files and confirmed 78,672 rows, 180,928 edges, exact graph/embedding feature equality, zero bio-hash mismatches, unchanged topology, edge attributes, labels, and user order, zero all-zero rows, and zero nonfinite values. The canonical SHA-256 fingerprints are:

- source CSV: `62cd481247b10d3d1c4dd792aa05d413f3ad01aa9331b91039b32ef7c98be42a`
- embedding: `83bfc1e105e789d865fb32150d852d6cbd7e3bab5b96903d1a69b3eb2a20e81c`
- graph: `755219daede78f913ab897fec51c98f261cff797a9869d80fb7ed35ce0b4d516`

Training and evaluation records must name the feature version when comparing results produced before and after this promotion.

The legacy snapshot is `/dataMeR1/phil/data/covid_political/legacy/cleaned-profile-v001/`. It contains the graph, graph sidecar, embedding, embedding sidecar, source `user_data.csv`, and a verified `SHA256SUMS` manifest.

## Election 2020

No `raw_profile` source was found in `social_llm_data/election2020`, its topology-only `graph.pickle`, or the searched Tucker data/project trees. The existing graph is therefore designated `cleaned-profile-v001`. Do not synthesize punctuation or digits or call a reconstructed string “raw.” A raw-profile replacement requires recovery from the original upstream collection.

Recovery checks also covered the original CARC location `/project2/emiliofe_74/julie/social_llm_data/election2020/`. Its `user_data.csv` and `graph.pickle` are byte-identical to Tucker (SHA-256 `0cb20c0f...304b` and `0fbfd1ec...3273`). The CSV has only `profile,label_conservative`; the graph and edge lists use positional integers and contain no account handle, Twitter user ID, description, or raw-profile attribute. The complete seven-commit history of the public `julie-jiang/retweet-bert` repository contains model code and links to public tweet-ID collections, but no deleted user map or profile data. Public rehydration cannot prove the historical 78,932-account mapping because this derivative discarded account identity. The required recovery input is the Retweet-BERT/Election preprocessing table containing account ID or handle alongside the original profile description and the final positional row assignment.

A read-only cross-dataset recovery audit tested whether later raw-bio stores can fill this gap. Election has 78,932 rows and 78,665 unique cleaned profile strings. Exact cleaned-profile matching against COVID Political `full_user_data.csv` recovers 17,698 Election rows; 17,600 of these map to exactly one handle and one raw profile. This is strong, auditable evidence for those rows, but covers only 22.30% of Election. It does not establish the missing 61,332 rows, so it is insufficient for a consistent replacement graph. A graph mixing raw-profile features for the recovered subset with cleaned-profile features elsewhere must not be treated as `raw-profile-v001`.

The recovered raw-profile hashes were also joined to the GTE bio observation stores for the larger graphs. Exact hash matches occur for 17,433 Election rows in COVID, 3,172 in Ukraine, and 1,402 in Midterm; respectively 17,328, 3,144, and 1,400 map to exactly one stored user ID. All three stores use `Alibaba-NLP/gte-multilingual-base`, revision `9bbca17d9273fd0d03d5725c7a4b0f6b45142062`, and preprocessing `bio-text-v001`. Their embedding rows are therefore valid reusable vectors for an exact normalized-bio hash. They do not extend full Election coverage: most are the same 17,600 profiles already recovered through COVID Political, and shared generic bios can map to multiple accounts.

Direct Election-cleaned-text searches of the large stores are much weaker. Exact normalized-text/hash matches cover 98 Midterm, 262 Ukraine, and 1,088 COVID rows. Applying the inferred historical cleaning operation to stored normalized bios raises coverage only to 100, 277, and 1,211 rows, with ambiguous user-ID matches. The inferred operation reproduces 128,431 of 128,441 COVID raw/clean pairs (99.992%): remove recognized URLs, delete ASCII digits, replace ASCII punctuation with spaces, and collapse whitespace. Because the large bio stores replace URLs and handles during `bio-text-v001` normalization, these transformed-text joins are lossy and are not identity evidence.

The complete counts and method notes are in `scripts/experiments/analysis/graphs/features/election_covid_alignment_audit/data/election_cross_graph_recovery_20260911.json`. Tucker and CARC source collections remained read-only throughout this audit.

The immutable reference snapshot is `/dataMeR1/phil/data/election2020/legacy/cleaned-profile-v001/`, with the graph, sidecars, embedding, source CSV, and verified `SHA256SUMS` manifest.

## Evidence

The full audit is in `scripts/experiments/analysis/graphs/features/election_covid_alignment_audit/`. Its principal findings are:

- all graph rows, embedding rows, user IDs, and stored bio hashes align exactly;
- upstream COVID cleaning changes the same-user embedding distribution by mean per-coordinate KS 0.0708;
- political selection and ideology mix are also material: matching Election and COVID Political label proportions reduces their mutual KS from 0.0669 to 0.0403–0.0526;
- re-encoding cleaned COVID profiles reproduces canonical vectors exactly.
- the later stores provide a verified raw-profile bridge for 17,600 Election rows, but not enough coverage to replace the canonical Election artifact.

The graph catalog carries the short operational warning so future loaders and agents encounter it at the registry entry rather than rediscovering it from geometry plots.
