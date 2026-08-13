# Findings: cross-dataset identity-overlap audit

## Scope and evidence hierarchy

The audit is aggregate-only and distinguishes literal account identifiers from
content proxies.

1. **Exact comparable Twitter IDs** are available for the full Ukraine, Covid,
   and Midterm graph-node universes.
2. **Partial exact IDs** are available for 56,440 of 72,295 Ukraine-Suspended
   nodes. Intersections involving this array are lower bounds for the complete
   target graph.
3. Covid-Political and Election store row indices, TwiBot-20 uses a dataset-
   internal namespace, and Hong Kong uses a hashed dataset namespace. Their
   stable-ID cells are explicitly `not_measurable`, not zero.
4. Facebook page IDs are a different platform namespace and are marked
   incompatible.
5. Exact graph-aligned `bio-text-v001` hashes provide a content-overlap proxy
   across all eight Twitter graphs. The strict proxy requires at least 20
   normalized characters and a hash that occurs once in each graph. It is not
   identity proof because normalization replaces URLs and handles and different
   accounts can reuse text.

No identifiers, handles, or biographies are serialized in the outputs.

## Exact-ID result

| Pair | Shared IDs | Coverage of A | Coverage of B | Jaccard |
|---|---:|---:|---:|---:|
| Ukraine--Covid | 4,294,139 | 41.29% | 18.66% | 14.75% |
| Ukraine--Midterm | 213,437 | 2.05% | 62.43% | 2.03% |
| Covid--Midterm | 185,539 | 0.81% | 54.27% | 0.80% |
| Ukraine--Ukraine-Suspended (partial) | 55,446 | 0.53% | 98.24% | 0.53% |
| Covid--Ukraine-Suspended (partial) | 32,201 | 0.14% | 57.05% | 0.14% |
| Midterm--Ukraine-Suspended (partial) | 7,194 | 2.10% | 12.75% | 1.84% |

These are exact intersections of reconstructed graph-node universes under the
original valid-retweet and timestamp filters. An independent `INTERSECT` query
against the frozen DuckDB tables reproduced 4,294,139 for Ukraine--Covid and
55,446 for Ukraine--Ukraine-Suspended. The full graph-node counts also reproduced
the catalog exactly: 10,400,775 Ukraine, 23,012,850 Covid, and 341,908 Midterm.

The collections are temporally separated but reuse many accounts: Covid spans
2020-01-23 to 2021-02-21, Ukraine 2022-02-22 to 2022-04-28, and Midterm
2022-10-01 to 2022-10-23. Exact overlap can therefore reflect persistent users
and high-visibility retweet targets rather than simultaneous collection alone.

## Biography-duplication proxy

Graph-aligned nonempty biography-hash coverage ranges from 76.7% to 100% across
the eight Twitter graphs. The largest strict unique-long intersections are:

| Pair | Shared nonempty hashes | Strict unique-long hashes |
|---|---:|---:|
| Ukraine--Covid | 1,529,364 | 1,235,004 |
| Ukraine--Midterm | 103,906 | 91,590 |
| Covid--Hong Kong | 91,797 | 69,686 |
| Ukraine--Hong Kong | 62,658 | 46,965 |
| Covid--Midterm | 57,172 | 46,204 |
| Covid--TwiBot-20 | 49,988 | 43,460 |
| Ukraine--Ukraine-Suspended | 37,196 | 32,241 |
| Covid-Political--Election | 16,334 | 16,208 |

The alternative SQL query independently reproduced 1,235,004 strict
Ukraine--Covid hashes. The large Covid-Political--Election proxy overlap is
especially important because stable IDs were discarded in both graph artifacts.

## Interpretation boundary

The paper can replace “identity overlap is unaudited” with “identity overlap is
quantified but incomplete.” It cannot claim that literal entity reuse is
irrelevant. Large exact and biography overlaps mean some off-diagonal specialist
scores and source--receiver associations may reflect repeated entities or
profile content in addition to transferable graph abstractions.

Removing the limitation would require comparable stable IDs for every Twitter
graph and entity-disjoint retraining or evaluation that removes overlapping
centers and sampled subgraph nodes. That experiment is not performed here.

## Frozen provenance

- Implementation: `27a1b10`; direct-execution fix: `c44720c`.
- Tucker worktree: `/dataMeR1/phil/gfm/prodigy-identityaudit`.
- Completed output: `state/identity_overlap_audit/v002`.
- Aggregate SHA-256:
  - inventory: `727f9ede35a3bde8d6ab5a3f256a91421ee48ed44b1aecd47bb9ab60854b353a`
  - exact-ID matrix: `f0ac15322d1eb839672f0c101a778e0f1d5e11b2057490c5665c744e855b6b4b`
  - biography matrix: `5c8c8401bafd604497ad64f0fb630e1b180df43469c80906cc880bbdb700988e`
  - summary: `cd21d9517986619fd80c107f221420cc97e100676f46f71856f7a504a8795da3`

The three CSVs are packaged with LF line endings; `summary.json` also retains
the original Tucker CRLF hashes. Values and row order are unchanged.
