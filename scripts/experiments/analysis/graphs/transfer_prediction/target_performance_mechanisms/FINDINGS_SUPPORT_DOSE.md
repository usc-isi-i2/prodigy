# Support suppression has opposite dose-response curves across targets

7 September 2026. Complete frozen-checkpoint replay; no new training.

## Contribution-level finding

Increasing support-edge suppression progressively improves Hong Kong political
ranking but progressively worsens bot ranking for both sources. Each direction
holds in all 18 individual seed/stream/mask curves for that source and target.
The effect is learned and training-stage dependent: on pages, removing the same
support edges hurts all six Hong Kong seed/stream comparisons at step 100 but
helps all six at step 2500. These controls establish more than a binary endpoint
contrast without implying a universal suppression policy or a solved source-data
cause. They complement the completed degree-preserving and message-content controls.

## Dose response at update 2500

Mean delta AUC in percentage points, relative to each model's intact inference.
Three mask draws are averaged within each of the six seed/stream evaluations.
Every cell, including negative and nonmonotone results, is retained.

| Target | Source | 25% | 50% | 75% | 100% | Nondecreasing / nonincreasing individual curves (of 18) |
|---|---|---:|---:|---:|---:|---:|
| covid-political | hongkong | +1.030 | +2.353 | +4.888 | +7.715 | 18 / 0 |
| covid-political | ukraine | -0.011 | -0.044 | -0.082 | -0.120 | 0 / 10 |
| election2020-political | hongkong | +0.040 | -0.008 | -0.037 | +0.379 | 5 / 0 |
| election2020-political | ukraine | +0.019 | +0.029 | +0.046 | +0.032 | 7 / 1 |
| facebook-page-reference | hongkong | +0.272 | +0.493 | +0.806 | +1.224 | 12 / 0 |
| facebook-page-reference | ukraine | +0.060 | +0.175 | +0.181 | +0.196 | 7 / 0 |
| twibot20 | hongkong | -0.882 | -2.401 | -4.277 | -8.851 | 0 / 18 |
| twibot20 | ukraine | -0.732 | -1.865 | -3.145 | -5.353 | 0 / 18 |
| ukraine-suspended | hongkong | +0.307 | +0.726 | +1.615 | +2.383 | 7 / 0 |
| ukraine-suspended | ukraine | +0.151 | +0.118 | +0.656 | +1.266 | 6 / 0 |

The endpoints are shared by the three nested draws, not three independently
estimated endpoints. Three initialization seeds remain the training replication
unit; draws and streams are not extra training seeds. Curves are nondecreasing
or nonincreasing at the tested doses, not assertions about all continuous doses.

Actual removed edge fractions differ slightly from requested edge-unit fractions
due to per-subgraph rounding and reciprocal-pair units. At the nominal 25/50/75%
doses, politics removes 25.77/50.34/75.78% of edges, pages approximately
25.98-26.03/52.23-52.28/76.71-76.78%. Complete removal is exactly 100%.
The 150-row actual-fraction table retains all target/stream/draw counts.

## Training-stage sensitivity

All available saved steps were included: 0, 100, 300, 900 and 2500. Source arms
share the exact initial weights within a seed, and their initial predictions
match. Political removal effects at initialization are below 0.01 AUC points.

| Target/source | Step 0 | Step 100 | Step 300 | Step 900 | Step 2500 |
|---|---:|---:|---:|---:|---:|
| Political / hongkong | -0.007 to +0.001 | +7.214 to +11.973 | -5.338 to +5.718 | +4.521 to +8.833 | +4.651 to +11.101 |
| Pages / hongkong | -0.001 to +0.005 | -1.929 to -0.141 | -3.248 to +2.677 | -0.109 to +0.723 | +0.885 to +2.035 |
| Bot / hongkong | -0.006 to +0.010 | -31.813 to -26.168 | -14.110 to -7.933 | -12.332 to -7.576 | -11.296 to -6.550 |
| Bot / ukraine | -0.006 to +0.010 | -34.581 to -7.572 | -8.680 to -4.106 | -6.160 to -4.404 | -6.265 to -4.723 |

Ranges are the six observed seed/stream effects, not confidence intervals.
Political repair recurs at multiple stages, but step 300 is explicitly mixed.
The page sign reversal holds in every matched seed and stream. Bot ranking is
harmed for both sources at all four nonzero saved steps. The remaining target/
source results are in `step_summary.csv`; no checkpoint was chosen using query
performance. Nothing here establishes behavior beyond the saved training window.

Probability loss is a separate outcome. At the final checkpoint all 18 Hong Kong
bot curves have both nonincreasing AUC and nonincreasing NLL: suppression worsens
ranking while improving probability loss. Political NLL is not monotone in every
draw even though political AUC is. AUC improvements should not be relabeled as
generic improvements of every metric.

## Design and verification

- 1140 metric cells; six existing production-policy controls, five targets,
  two cached streams of 128 binary episodes each; 30 saved model states.
- Intact and support-only removal at all five steps; three shared nested mask
  draws at 25/50/75% suppression additionally at step 2500.
- Label-blind within-subgraph permutations; reciprocal edge pairs stay together;
  self loops are removed at the complete-removal endpoint. Features, members,
  pooling connections and all query inputs remain fixed.
- All 300 model/step/target/stream receipts pass; 26,880 query-encoding checks
  before and 26,880 after the metagraph are bit-exact. Weights and buffers remain
  unchanged. Every final intact/removal endpoint matches the prior topology grid.
- Completed counters are checked against training-state sidecars; initial and
  terminal digests match verified arms. Initial source-weight and metric parity
  pass. All 4800 mask records and the complete declared grid are validated.
- CPU-only runtime `e664d220`, 1532.977 seconds. Summary/draw audit `a6b22396`.
  Full outputs: `/dataMeR1/phil/gfm/prodigy-role-topology/log/support_dose_full_20260907`.
  Per-query prediction tensors remain there; compact JSON receipts and CSVs are
  committed under `data/support_dose_20260907/` with source-file hashes.
- Branch `codex/role-topology-interactions`; local worktree
  `/Users/philipp/projects/gfm/prodigy/.worktrees/role-topology`; isolated Tucker
  worktree `/dataMeR1/phil/gfm/prodigy-role-topology`.

## Reproduction and publication consequence

Run `analyze_support_dose --input <completed-json-folder> --output <new-folder>`
and `plot_support_dose --input <csv-folder> --output <figure-folder>` as modules
under `scripts.experiments.analysis.graphs.transfer_prediction.target_performance_mechanisms`.
Use the local Homebrew Python 3.11 and `MPLBACKEND=Agg` for plots. The complete
validator rejects smoke outputs, incomplete grids and failed frozen-input checks.

The manuscript should state the opposite dose-response and within-model stage
reversal plainly. The review's dose and checkpoint requests are now completed;
do not keep rerunning these grids. A portable inference release and a more direct
source-training/annotation-cue explanation are separate remaining work. This
study strengthens an actionable role-specific diagnosis, not an automated best
intervention selector or a causal account of every source-performance difference.
