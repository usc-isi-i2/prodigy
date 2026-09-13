# NM broad native/foreign stage audit

## Scope and provenance

This audit replays the **published randomized fixed-test seed-0 episodes** for
Ukraine/Russia, COVID, Midterm, and Hong Kong against all four single-source
specialists.  Each cell contains 61,440 query occurrences (512 episodes).  It is
therefore a consistent 4 x 4 extension of the native/foreign stage question, but
it is a different episode protocol from the detailed lowest-sorted HK/Ukraine
canonical audit and must not be pooled with it.

The replay passed all acceptance gates: the four historical raw-plan and observed
input fingerprints matched, every one of the 16 final accuracies matched the
published result exactly, checkpoint state hashes were unchanged, and no training
or new episode sampling occurred.  Model replay took 949.5 seconds on one GPU;
the cached aggregate is CPU-only.

## Result

| target | model | pre-M acc. | final acc. | net readout | wrong-to-correct | correct-to-wrong |
|---|---|---:|---:|---:|---:|---:|
| Ukraine/Russia | **Ukraine/Russia** | **38.89%** | **44.15%** | +5.26 pp | 6,714 | 3,485 |
| Ukraine/Russia | best foreign (COVID) | 34.94% | 38.41% | +3.47 pp | 5,808 | 3,679 |
| COVID | **COVID** | **53.62%** | **58.08%** | +4.46 pp | 6,349 | 3,610 |
| COVID | best foreign (Ukraine/Russia) | 50.35% | 55.33% | +4.99 pp | 6,853 | 3,788 |
| Midterm | **Midterm** | **23.46%** | **25.86%** | +2.39 pp | 6,153 | 4,682 |
| Midterm | best foreign (COVID) | 20.50% | 21.47% | +0.97 pp | 4,745 | 4,152 |
| Hong Kong | **Hong Kong** | **12.98%** | **17.86%** | +4.89 pp | 6,139 | 3,135 |
| Hong Kong | best foreign (Ukraine/Russia) | 10.88% | 11.68% | +0.80 pp | 3,468 | 2,976 |

Source matching improves pre-metagraph separation on every target relative to its
best foreign comparator.  The size and location of the advantage vary:

- Ukraine/Russia and COVID already have substantial native pre-M advantages.
- Midterm has both a pre-M advantage and a larger native readout gain.
- Hong Kong has only a 2.10-point pre-M advantage over its best foreign model, but
  a 4.09-point advantage in net readout gain.  Its final native advantage is thus
  predominantly created after support labels enter through the metagraph.

The paired native-only cohorts agree with this interpretation.  For HK queries
solved only by the native model, just 32.7--35.1% are already correct at pre-M,
depending on the foreign comparator; roughly two thirds of these native-only final
successes are created by the readout.  The analogous pre-M fractions against each
target's strongest foreign model are 51.2% for Ukraine/Russia, 53.6% for COVID,
and 45.6% for Midterm.

The metagraph is not simply harmful.  Every one of the 16 cells has positive net
readout accuracy.  It also changes many decisions in both directions: even the
native HK model rescues 6,139 pre-M errors while breaking 3,135 pre-M successes.
This recover/loss tradeoff is common across native models.  What is unusually large
for HK is the *source-specific difference* in net readout benefit.

## What this establishes

This is a controlled computational localization on identical target episodes.
It establishes that source-dependent NM performance is not explained by one
universal stage: source training changes pre-M ordering on all four targets, while
the amount converted by the support-conditioned readout is strongly target and
source dependent.  HK is the clearest readout-dominant case in this panel.

It does not establish which learned parameter, support statistic, or training event
causes those stage differences.  The checkpoints were trained independently, so
encoder and metagraph parameters co-vary.  It also does not turn pre-M scores into a
deployable gate: fixed blending already failed in the detailed HK audit, and every
model loses some correct pre-M decisions during readout.

## Implication for the next model test

The most informative bounded fix is a learned, episode-conditioned residual gate
between pre-M evidence and the metagraph update, trained on ordinary NM episodes and
evaluated with wrong-to-correct and correct-to-wrong counts.  The gate should use
signals available at inference (pre-M margin/rank shape, support agreement, and
metagraph update magnitude), rather than known true labels or selected supports.
Before training it, a cached oracle/feature audit should measure how much of the
recover-versus-break choice is predictable across all four native graphs.  That
determines whether gating has defensible headroom and whether its predictors
generalize beyond HK.

## Artifacts

- Compact aggregate: `data/canonical_split/nm_broad_stage_20260909_v3.json`
- Aggregator: `aggregate_nm_broad_stage.py`
- Private complete inputs, stage tensors, logits, and row tables:
  `/dataMeR1/phil/gfm/error_audit/nm_broad_stage_20260909_v3/`

