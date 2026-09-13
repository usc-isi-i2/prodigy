# Native-graph replication: HK and Ukraine NM class references

9 September 2026. Frozen HK- and Ukraine-trained checkpoints replayed on both
canonical targets, always using identical cached episode inputs within a target.
Each of the four cells contains 512 episodes and 61,440 query occurrences. No
graph loading, sampling, encoding, training, or GPU use.

## Positive-support dominance is general, not an HK-only failure mode

| Target | Model | Accuracy | Errors | Positive favors wrong winner | Positive largest pro-error component |
|---|---|---:|---:|---:|---:|
| HK | HK native | 17.86% | 50,464 | 93.39% | 77.76% |
| HK | Ukraine foreign | 11.68% | 54,263 | 91.89% | 86.03% |
| Ukraine | Ukraine native | 44.15% | 34,317 | 92.22% | 82.50% |
| Ukraine | HK foreign | 18.15% | 50,288 | 91.63% | 79.30% |

The positive-support term carries the wrong-winner direction in nearly the same
fraction of errors in every cell. This is a structural property of how the NM
model forms class references and scores queries. It cannot by itself explain
why HK→HK is weak or why native source training helps.

The informative quantity is how often that path points to truth. Positive-only
accuracy is 42.01% for Ukraine→Ukraine, versus 17.05% for HK→HK, 17.01% for
HK→Ukraine, and 10.37% for Ukraine→HK. Adding negative-support terms changes
these to 42.88%, 16.94%, 17.49%, and 11.41%. Negative messages are secondary in
all four cells and do not account for the large native-Ukraine advantage.

## Source divergence begins at different stages

| Target | Stage | Native | Foreign | Native advantage |
|---|---|---:|---:|---:|
| HK | Pre-metagraph cosine | 12.98% | 10.88% | +2.09 pp |
| HK | Final | 17.86% | 11.68% | +6.18 pp |
| Ukraine | Pre-metagraph cosine | 38.89% | 17.05% | **+21.84 pp** |
| Ukraine | Final | 44.15% | 18.15% | **+25.99 pp** |

Ukraine's native advantage is mostly present in the learned query/support
geometry before the metagraph. The final readout adds another 4.15 points of
separation. HK has little native separation before the metagraph; its readout
amplifies that smaller advantage by 4.09 points.

This rules out one universal source-transfer account located only in the final
metagraph. Source training changes pre-metagraph representations strongly on
Ukraine, while the HK native/foreign distinction is more dependent on the
support-derived final decision.

## Paired winner direction replicates

On Ukraine there are 18,881 Ukraine-only successes and 2,910 HK-only successes.
For Ukraine-only cases, Ukraine's positive term favors truth on 88.60% with
mean rival-minus-truth gap −1.499, while HK's favors its wrong winner on 89.70%
with mean +1.546. For HK-only cases, HK's mean is −0.501 and Ukraine's is
+1.168. This mirrors the HK target: whichever source model uniquely succeeds
usually differs in the learned positive-support reference direction.

Both models fail on 31,407 Ukraine occurrences, selecting the same wrong class
in only 4,955. As on HK, shared failure does not mean attraction to a common
distractor.

## Consequence for the training-conflict hypothesis

The existing consumed-training audit reports query/support identity conflicts
at 22.22% for HK and 2.10% for Ukraine. Native NM accuracy is correspondingly
17.86% versus 44.15%, and Ukraine has far stronger pre-metagraph separation.
This is a source-level correlation consistent with harmful exclusive
supervision in HK. Two sources do not establish causality, and the prior
member-policy controls changed several properties at once.

The proposed matched HK training intervention remains discriminating, but its
claim must be narrower: test whether contradicted support labels cause part of
HK's missing pre-metagraph/reference separation. Positive-reference dominance
is the readout pathway through which errors appear, not itself the source-specific
cause. A successful intervention must improve multi-positive and uniquely
answerable HK NM while preserving successes; a change in component gaps alone
is insufficient.

[HK full decomposition](FINDINGS_NM_HK_CLASS_REFERENCE_FULL.md) ·
[Ukraine paired aggregate](data/canonical_split/nm_ukr_native_foreign_references.json) ·
[HK paired aggregate](data/canonical_split/nm_hk_native_foreign_references.json).
Private per-occurrence decompositions remain under `/private/tmp`. Ukraine
replays took 17.71 and 17.07 seconds on two local CPU threads; no GPU was used.
