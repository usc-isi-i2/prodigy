# Findings: exact-ID-clean episode centers

## Design

The diagnostic reuses the frozen step-2500 PRODIGY specialists and original
30-way, 3-shot fixed-test protocol. For each target, it excludes every identity
that occurs in either of the other two full exact-ID graphs. The same clean
target episodes are used for both off-diagonal donors and all three training
seeds. Diagonal cells and target-containing mixtures are undefined under this
control and are not reported.

The preflight exactly reproduced each original target plan fingerprint before
scoring. The clean run contains 18 physical cells: six directed source--target
pairs by three seeds.

## Result

Across the 18 cells, mean accuracy changes from **0.3192** on the original fixed
streams to **0.3148** on exact-ID-clean episode centers, a **-0.0044** absolute
change. Transfer therefore does not collapse in aggregate, but the change is
strongly directional and consistent across seeds within five of six pairs.

| Source -> target | Original | Center-clean | Delta |
|---|---:|---:|---:|
| Ukraine -> Covid | .5557 | .4820 | -.0737 |
| Ukraine -> Midterm | .2145 | .1869 | -.0276 |
| Covid -> Ukraine | .3844 | .4262 | +.0418 |
| Covid -> Midterm | .2136 | .1938 | -.0199 |
| Midterm -> Ukraine | .2166 | .2729 | +.0564 |
| Midterm -> Covid | .3302 | .3271 | -.0032 |

These are comparisons across two fixed target distributions, not paired-episode
effects: the original and center-clean episode sets differ by construction.
They show sensitivity to recurring target identities while also showing that
nontrivial transfer remains after episode-level identity removal.

## Remaining contamination and decision

Episode anchors and every support/query center are exact-ID-clean, but encoder
subgraphs still use the original static-train background. Residual overlapping
identity occurrences constitute 55.6% of sampled Ukraine context, 36.7% of
Covid context, and 30.6% of Midterm context (741,996; 666,362; and 77,297 unique
overlapping nodes respectively). Therefore this stage is not evidence for fully
entity-disjoint graph abstraction.

The diagnostic warrants the stronger second stage: rebuild message-passing
samplers on the allowed-node induced subgraphs and rescore the same 18 cells.
Only that result can address repeated identities in sampled encoder context.

## Provenance

- Implementation commit: `324fc03` plus the aggregate-documentation follow-up.
- Branch/worktree: `codex/entity-disjoint-eval`,
  `/dataMeR1/phil/gfm/prodigy-entitydisjoint`.
- Run: `state/entity_disjoint_eval/center_clean_v001`.
- Completed UTC: `2026-08-12T06:10:17Z`.
- Paired table SHA-256 before the denominator-column regeneration:
  `e3dd1177830deb9ef79c15ef0530d47c2604b06674a15091fc0534dba8908129`.
- Frozen original, clean-plan, and observed-stream fingerprints are stored in
  `data/center_clean_paired_cells.tsv` and `data/center_clean_summary.json`.
