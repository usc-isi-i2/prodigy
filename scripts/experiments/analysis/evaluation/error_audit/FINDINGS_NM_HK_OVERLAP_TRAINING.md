# HK→HK NM: matched overlap-aware training intervention

9 September 2026. One fixed-seed, 2,500-update baseline/treatment pair tests
whether suppressing training-view-contradicted negative support messages repairs
HK neighbor matching. Both arms use the same initialization and exactly the same
realized episodes; the treatment changes only those metagraph messages. All
checkpoint evaluations replay the same 61,440 cached canonical queries. The
checkpoint schedule was fixed before training and no test result selected a
checkpoint.

## Integrity and compute

The verifier confirms exact equality of all 2,500 consumed episode inputs and
an identical initial model hash. Terminal model hashes differ. The treatment
masked 4,694,600 negative messages (mean 1,877.84 per update; range
1,315–2,631), with a nonempty mask in every update.

Actual compute was 80.0 minutes for baseline training and 101.7 minutes for
treatment training, sequentially on physical Tucker GPU 0 at low CPU priority.
The priority VISION jobs ran on other GPUs. Canonical evaluation took 32.8
minutes on two low-priority CPU threads and no GPU. The earlier setup estimate
was 20–30 minutes; setup and smoke validation fit that range. The earlier
15–20-minute-per-arm training estimate was too low because zero-worker sampling
was heavily CPU-bound under concurrent priority jobs.

## Result

| Step | Baseline multi-positive | Treatment multi-positive | Change | Baseline unique | Treatment unique | Change |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 6.20% | 6.20% | 0.00 pp | 3.30% | 3.30% | 0.00 pp |
| 100 | 9.89% | 10.37% | +0.48 pp | 7.87% | 8.08% | +0.21 pp |
| 300 | 15.67% | 16.07% | +0.41 pp | 15.56% | 15.64% | +0.08 pp |
| 900 | 22.12% | 20.89% | −1.23 pp | 21.10% | 20.02% | −1.08 pp |
| 2,500 | **23.75%** | **21.96%** | **−1.79 pp** | **22.51%** | **21.19%** | **−1.32 pp** |

At the fixed terminal step, assigned-anchor accuracy also falls from 18.15% to
16.53% (−1.63 points). The treatment recovers 5,303 baseline multi-positive
failures but breaks 6,404 baseline successes, a net loss of 1,101 query
occurrences. On uniquely answerable rows it recovers 2,315 and breaks 2,751, a
net loss of 436. At episode level, 156/512 improve, 36 tie, and 320 worsen; the
median multi-positive change is −1.67 points. The negative result therefore is
not confined to ambiguous labels or a small number of episodes.

The small early gains reverse between steps 300 and 900. They show that the
message change alters learning in the intended pathway, but the terminal result
fails the predeclared repair criterion: corrected and unique-anchor accuracy
both decline, and lost successes exceed recovered failures.

## Interpretation

This is controlled evidence against the narrow claim that HK failure is repaired
by simply withholding every known training-neighbor negative message. Combined
with the earlier inference-time mask, which was also slightly harmful, it rules
out contradicted-message deletion as a sufficient targeted fix at either
inference or training time.

It does not show that many-to-many label conflict is irrelevant. The intervention
changes only support-to-label messages, retains the exclusive single-anchor
cross-entropy target, and removes a large fraction of the normal negative
context. It may therefore reduce useful contrast while leaving the contradictory
supervision itself intact. One fixed seed also cannot establish a precise mean
effect across training randomness, although the exact paired stream makes the
direction within this run unambiguous.

The most informative next mechanism is source-distribution coverage at the
pre-metagraph stage, using fixed target inputs and native/foreign checkpoints.
For each target query and support subgraph, measure whether native-only successes
occupy regions better covered by the source training distribution, then test a
matched source-coverage intervention or representation regularizer. This follows
the established stage split: Ukraine's native advantage is already mostly
pre-metagraph, whereas HK's smaller pre-metagraph advantage is amplified by the
readout. Another support-deletion heuristic is not warranted by these results.

[Canonical evaluation receipt](data/canonical_split/nm_hk_overlap_training_pair.json) ·
[matched-stream verification](data/canonical_split/nm_hk_overlap_training_pair_verification.json) ·
[protocol](../../../setup/nm_hk_goal/README.md).
