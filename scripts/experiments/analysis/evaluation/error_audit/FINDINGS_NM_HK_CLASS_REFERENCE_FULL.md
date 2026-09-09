# HK→HK NM: full class-reference and native/foreign decomposition

9 September 2026. Exact replay of all 512 canonical HK test episodes: 61,440
query occurrences and 50,464 assigned-anchor errors. The native HK and foreign
Ukraine checkpoints were evaluated on identical cached episode inputs. The run
loaded cached pre-metagraph tensors and frozen 2,500-update checkpoints; it did
not load a graph, sample, encode, train, or allocate a GPU.

## The first episode generalizes across HK

For every query, the final rival-minus-truth score is decomposed exactly into
the realized positive-support, negative-support, label-self, output-bias,
label-residual, and batch-normalization terms. Reconstruction error is at most
`1.43e-5`, no native argmax differs from the canonical record, and the model
state is unchanged.

Across all 50,464 native HK errors:

| Component result | Errors | Fraction |
|---|---:|---:|
| Positive-support term favors the wrong winner | 47,130 | **93.39%** |
| Positive-support term is the largest rival-favoring component | 39,240 | **77.76%** |
| Positive-support gap alone exceeds the complete error gap | 25,479 | 50.49% |
| Negative-support term favors the wrong winner | 18,897 | 37.45% |

The mean rival-minus-truth contribution is +1.609 from positive supports,
−0.159 from negative supports, +0.224 from output bias, and close to zero from
the remaining terms. Negative messages therefore oppose the chosen error on
average. They are not the dominant immediate cause of most realized failures.

This is not an artifact of the first, below-median episode. Across the 512
episodes, the fraction of errors whose positive term favors the wrong winner
ranges from 82.29% to 99.04% (median 93.72%). The fraction where it is the
largest rival-favoring component ranges from 59.57% to 92.55% (median 77.89%).

Repeated queries do not create the result. The 50,464 errors contain 4,987
distinct query nodes. Equal weighting over nodes gives a mean 89.12% of each
node's failed occurrences with a positive term favoring the winner; 89.59% of
error nodes have this pattern in a majority of their failed occurrences.

## Correct controls separate sharply

The same calculation on all 10,976 native successes points in the opposite
direction. The positive-support term favors truth over the strongest rival in
82.02% of correct occurrences, with a mean rival-minus-truth contribution of
−0.979. On errors it favors the wrong winner in 93.39%, with mean +1.609.

The separation survives the evaluation correction:

| Population | Occurrences | Positive term favors wrong rival | Mean positive gap |
|---|---:|---:|---:|
| Uniquely answerable correct | 7,216 | 17.09% | −1.251 |
| Uniquely answerable error | 25,817 | **92.28%** | +1.732 |
| Ambiguous assigned correct | 3,760 | 19.68% | −0.457 |
| Ambiguous assigned error | 24,647 | **94.56%** | +1.481 |
| Wrong under multi-positive scoring | 47,149 | **93.29%** | +1.617 |

Thus positive-reference misalignment describes genuine candidate-set errors as
well as single-label evaluation artifacts. It does not identify which support
property produced the direction.

Among the 43,960 errors unseparated by pre-metagraph cosine, raw neighborhood
mean, or sampled-node overlap, the positive term favors the wrong winner in
94.42% and is the largest component in 79.47%. This localizes the largest hard
population to the learned support-to-class-reference decision, conditional on
the representations already supplied to it.

## Fixed-reference score ablations are not a repair

Using only the intact positive-support score produces 17.05% accuracy: it
recovers 1,966 native errors but loses 2,468 native successes. Adding the
negative-support term produces 16.94%, recovering 2,060 and losing 2,631.
Positive support carries most of the decision, but deleting other learned terms
does not improve the benchmark. This mirrors the earlier support-replacement
and negative-mask results: interventions can rescue failures while breaking
more controls.

## Native versus foreign models on identical HK episodes

Input evidence is held fixed. Divergence is already present before the
metagraph, then widens at the final decision:

| Stage | HK model | Ukraine model | Native advantage |
|---|---:|---:|---:|
| Pre-metagraph mean cosine | 12.98% | 10.88% | +2.09 pp |
| Final model | 17.86% | 11.68% | **+6.18 pp** |

The HK metagraph rescues 6,139 pre-metagraph misses and loses 3,135 successes;
Ukraine rescues 3,468 and loses 2,976. Hence the readout is useful for both, but
creates much more net benefit for the native model.

Final outcomes contain 3,411 shared successes, 7,565 HK-only successes, 3,766
Ukraine-only successes, and 46,698 shared errors. On HK-only successes, the HK
positive term favors truth on 79.18% with mean gap −0.582, while Ukraine's term
favors its wrong winner on 89.80% with mean +0.849. On Ukraine-only successes,
the direction reverses: Ukraine mean −0.380 versus HK +1.121. This places the
source-dependent boundary primarily in each checkpoint's learned positive
support references, rather than a common final bias.

Both models still fail on 46,698 occurrences. They select the same wrong class
on only 7,228 of these; overall prediction agreement is 17.32%. Their shared
difficulty is therefore not one universal distractor. On shared failures the HK
positive gap is larger (+1.649 versus +1.048), while its final accuracy is still
higher. Gap magnitude cannot by itself rank source models.

## Evidence status and next mechanism test

This is an exact population-wide localization result and a fixed-input
cross-model comparison. It establishes that the positive-support class-reference
path carries most correct and incorrect decisions and is where the native/foreign
winner direction usually differs. It is not a causal account of why source
training learned those references: each component uses the realized normalized
reference and shares upstream attention and normalization.

Together with the consumed-training audit, the leading hypothesis is now
specific: HK's heavily overlapping neighborhoods, deterministic low-ID role
assignment, and exclusive labels train positive class references that track
recurrent semantic/high-degree aliases rather than exact candidate adjacency.
Training conflict is directly observed, and final positive-reference dominance
is directly observed, but the link between them remains unisolated.

The next informative intervention is a matched-input HK training pair that
changes only contradicted support-to-rival labels. Before training, compare
per-example gradients on saved conflicting and nonconflicting batches. Then
evaluate both checkpoints on the same canonical episodes, with multi-positive
top-1 primary and uniquely answerable accuracy, recoveries, and lost successes
as required controls. This tests the proposed training link without changing
source coverage, centers, members, sampled contexts, or the query target.

[Native aggregate](data/canonical_split/nm_hk_class_reference_full.json) ·
[native/foreign aggregate](data/canonical_split/nm_hk_native_foreign_references.json) ·
[training-conflict bridge](FINDINGS_NM_HK_TRAINING_CONFLICT_BRIDGE.md).
Private per-occurrence decompositions remain under `/private/tmp` and are not
committed. The two successful replays took 17.94 and 17.59 seconds on two local
CPU threads; no GPU was used.
