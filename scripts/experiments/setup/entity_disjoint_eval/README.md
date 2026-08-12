# Exact-ID center-disjoint evaluation

This control reuses the frozen 93 PRODIGY final-core checkpoints and the fixed
30-way, 3-shot neighbor-matching protocol. It does not retrain models.

For each of Ukraine, Covid, and Midterm, the target evaluation pool excludes
every identity whose platform-global Twitter ID also occurs in either of the
other two graphs. This **union-three** rule produces one common clean episode
stream per target. The diagnostic evaluates the three specialists in the six
off-diagonal directions across all three existing seeds (18 cells). Diagonal
cells and models trained on the target are undefined under entity-disjoint
evaluation and are not reported.

The first diagnostic filters all episode-level identities: the 30 anchor centers
and all support/query node centers. Sampled two-hop encoder context can still
contain recurring identities; its overlap occurrences and unique-node count are
audited in every result record. A second-stage induced-subgraph filter should run
only if the episode-center-clean result survives.

Internal exclusion `.pt` files contain global graph-row indices but no user IDs.
Only aggregate counts, hashes, and performance differences belong in the
analysis tree or manuscript.

## Tucker inputs

- graph: `/dataMeR1/phil/data/merged/graphs/ukr_rus_covid_midterm_all9_facebook_final_core_split_seed0.pt`
- identity database: `/dataMeR1/phil/gfm/prodigy-identityaudit/state/identity_overlap_audit/v002/identity_overlap.duckdb`
- checkpoints: `/dataMeR1/phil/gfm/prodigy-final-core/state/final_core`

## Run

From the isolated Tucker worktree, after verifying GPUs 0 and 1 are free:

```bash
RUN_ID=center_clean_v001 \
  bash scripts/experiments/setup/entity_disjoint_eval/run_center_disjoint_tucker.sh
```

The launcher refuses occupied GPUs, requires at least 380 GiB available host
RAM for two persistent graph copies, never overwrites the exclusion directory,
and resumes already validated cell JSONs. Resumed exclusions must match the
current graph and identity-database hashes, comparison sources, and target-node
counts. Before the first GPU score, the evaluator verifies that the unfiltered
prefix exactly reproduces every relevant frozen original episode-plan
fingerprint.
