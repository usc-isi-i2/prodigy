# Findings: exact-ID-disjoint transfer

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

## Center-clean result

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

This motivated the stronger second stage below.

## Fully induced result

The second stage rebuilds both the static-train message-passing adjacency and
the static-test positive adjacency on the allowed target nodes. Across
16,613,455 sampled node occurrences, the evaluator observed **zero forbidden
nodes**. The exclusion is also a substantial domain intervention: it removes
41.8% of Ukraine nodes, 18.8% of Covid nodes, and 70.4% of Midterm nodes. Each
target plan and sampled stream was identical across the evaluated donor/seed
checkpoints.

| Source -> target | Original | Center-clean | Fully induced | Induced - original |
|---|---:|---:|---:|---:|
| Ukraine -> Covid | .5557 | .4820 | .4500 | -.1057 |
| Ukraine -> Midterm | .2145 | .1869 | .1653 | -.0492 |
| Covid -> Ukraine | .3844 | .4262 | .3877 | +.0033 |
| Covid -> Midterm | .2136 | .1938 | .1711 | -.0426 |
| Midterm -> Ukraine | .2166 | .2729 | .2429 | +.0264 |
| Midterm -> Covid | .3302 | .3271 | .2998 | -.0304 |
| **Mean** | **.3192** | **.3148** | **.2861** | **-.0330** |

The fully induced mean is 2.87 percentage points below center-clean and 3.30
points below the original streams. Nevertheless, transfer remains far above
30-way chance (3.33%) in every direction. Two of six directions remain above
their original accuracy, and the other four fall by 3.0--10.6 points.

The correct conclusion is therefore narrower than “identity overlap explains
transfer” or “identity overlap is irrelevant.” Removing literal recurring
identities is associated with lower scores in some directions, especially
Ukraine to Covid, but nontrivial cross-graph transfer remains. This is a
distributional comparison: the original, center-clean, and fully induced
episode streams differ, so their deltas are not paired-episode causal effects.

This control applies only to the three graphs with compatible complete Twitter
IDs. It does not establish entity-disjoint transfer for graphs whose identifiers
are local, hashed with incompatible provenance, partial, or cross-platform.

## Provenance

- Implementation commit: `324fc03` plus the aggregate-documentation follow-up.
- Branch/worktree: `codex/entity-disjoint-eval`,
  `/dataMeR1/phil/gfm/prodigy-entitydisjoint`.
- Run: `state/entity_disjoint_eval/center_clean_v001`.
- Completed UTC: `2026-08-12T06:10:17Z`.
- Final paired table SHA-256:
  `cca09927def35f2cfabf6c2e4759588f9555a9602bc6a1b7f254c25201ec5c3a`.
- Final aggregate summary SHA-256:
  `bb3f933de1de0f054e861faa66a7dcef4686217594200d7af8558f45f0f1dfec`.
- Frozen original, clean-plan, and observed-stream fingerprints are stored in
  `data/center_clean_paired_cells.tsv` and `data/center_clean_summary.json`.

Fully induced stage:

- Implementation-only branch: `codex/entity-disjoint-induced-run`, commits
  `29c642b` and `91e4b4b`; result-documentation branch remains local.
- Run: `state/entity_disjoint_eval/induced_disjoint_v001`.
- Completed UTC: `2026-08-12T08:28:35Z`.
- Paired table SHA-256:
  `6161c64d6e15f44c4cdb743aff376b1ecba40be02a17a6c30ae9b43eefa577b4`.
- Aggregate summary SHA-256:
  `401a1bae83837dafa55b03085ba464027bc6db23b297c342f71d15a414c4d611`.
- Frozen plan, observed-stream, adjacency-size, and zero-overlap audits are in
  `data/induced_paired_cells.tsv` and `data/induced_summary.json`.
