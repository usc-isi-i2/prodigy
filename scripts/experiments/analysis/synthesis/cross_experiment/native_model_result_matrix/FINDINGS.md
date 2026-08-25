# Native-model result-matrix audit

Last updated: 2026-08-25.

## Coverage snapshot

| model/version | SSL→CLS saturation | cross-SSL matrix | downstream CLS matrix | mixture diversity→CLS | adaptation efficiency |
|---|---|---|---|---|---|
| PRODIGY final-core 2.5k | complete | complete | complete | complete | complete |
| VISION native feature similarity | complete | complete | complete | running | complete |
| GILT native SSL | missing | missing | missing | missing | missing |
| SAMGPT native GraphCL | complete | complete | complete | partial | complete |
| GraphSAGE pilot-v1 | complete | missing | narrow TwiBot probe only | missing | complete |
| MLP | N/A | N/A | supervised reference complete | N/A | complete |
| raw logistic regression | N/A | N/A | raw-feature reference complete | N/A | complete |

This table distinguishes a complete registered design from a broad but
unregistered pile of files. In particular, SAMGPT downstream CLS is now complete
for its registered final-core design: 31 physical source-mixture models by nine
targets by three training seeds. VISION now has valid native single-source,
all-nine, and cross-SSL evidence; the registered mixture-diversity run is the
only active VISION gap.

## Evidence recovered before launching new work

- PRODIGY final-core all-nine and specialist checkpoints exist at step 2,500 for
  all three seeds in Tucker's final-core runtime archive.
- The complete family-native final-core table contains 1,944 logical cells:
  PRODIGY/NM and SAMGPT/GraphCL each have a three-seed 9×9 specialist matrix
  and three nine-rung ladders.
- SAMGPT has 93 native GraphCL physical runs with checkpoints at 20, 60, 180,
  and 500 updates. Its registered downstream table has 279 aggregate cells,
  corresponding to 31 models × 9 targets with all three training seeds.
- VISION's native source sweep completed the five-source × five-target × five
  checkpoint design at seed 0 (125 learned-checkpoint rows, plus five random-init
  controls). Its valid trainer uses label-free feature similarity. The 260-row
  physical export count includes both VISION and the excluded GILT sweep.
- Upstream GILT does have a genuine GraphCL implementation at commit
  `ba46cf4ebd1931712854708c221eaba646641785`: two augmented graph views are
  trained with NT-Xent/InfoNCE. The upstream checkout contains no saved model
  artifact, and the existing social-graph wrapper instead uses target-label
  episodic classification. GILT therefore remains missing rather than
  “native SSL undefined”; producing it requires a new social-graph port and
  native GraphCL training run.
- GraphSAGE pilot-v1 has a valid all-graph checkpoint after 2,000 native
  link-prediction updates. The existing TwiBot full-train node probe is retained
  as a narrow downstream reference, not treated as the requested adaptation
  grid.
- The missing GraphSAGE prefix was reconstructed at its recorded source commit
  and seed for 0/20/60/100/300/900/2,000 updates. Repeating the 2,000-update run
  reproduced every registered checkpoint tensor exactly (`max_abs_diff=0`,
  state SHA-256
  `cbca0b2ab6bf9eb0707f90ef2bf4073caf89da14460e7466cf326068f672f72f`).
  The prefix is therefore exact enough for matched downstream CLS evaluation.
  A narrow official-split TwiBot full-label probe across the exact prefix is now
  registered: ROC-AUC is 0.7617 at initialization and 0.7600 at 2,000 updates
  (−0.0017), while accuracy changes from 0.6974 to 0.7143. The nearly flat AUC
  means this native link-prediction run does not demonstrate improved bot-label
  ranking on the single available target. This is one training seed and uses all
  official training labels, so it is not adaptation-efficiency evidence.
- Existing raw-feature logistic and target-supervised MLP references are useful
  floors, but their earlier label budgets and optimization schedules are not the
  matched adaptation protocol.

## Work launched

Two new VISION all-nine native feature-similarity replicas were launched on
physical Tucker GPUs 2 and 3 with seeds 1 and 2. They use the same 2,500-update
fixed-compute schedule and checkpoints at 100, 300, 900, and 2,500 as seed 0.
Downstream trajectory evaluation used the repository's fixed episodes and
preserved all four checkpoints rather than only the terminal result. The final
export has 60/60 logical cells (3 seeds × 4 checkpoints × 5 targets), and every
target has one identical episode fingerprint across seeds and checkpoints.

VISION is already saturated at the first saved checkpoint in the five-target
mean: ROC-AUC is 0.7723 at 100 updates, 0.7689 at 300, 0.7631 at 900, and 0.7620
at 2,500. The terminal-minus-100 change is target-dependent: COVID Political
−0.0665, Election 2020 −0.0364, TwiBot-20 +0.0052, Facebook Page +0.0094,
and Ukraine Suspended +0.0367. More fixed compute changes which targets benefit;
it does not improve the registered panel mean.

A native VISION mixture-diversity run is active on Tucker GPUs 2 and 3. It uses the
three final-core source orders at rungs 1/3/5/7/9, deduplicating to 13 source
sets and reusing the existing all-nine seed-0 checkpoint. The 12 genuinely
missing models retain checkpoints 100/300/900/2,500 and the identical five CLS
episode streams. No mixture cell is credited until all 260 physical
model/checkpoint/target cells validate.

The VISION cross-SSL replay is complete and remains separate from CLS. It uses
128 deterministic label-free feature-similarity pseudo-episodes for five
existing specialists × five target graphs × checkpoints 20/60/100/300/900
(125/125 cells), recording pseudo-task accuracy and native SSL loss with one
fixed fingerprint per target. Mean pseudo-task accuracy across all 25
source→target pairs rises monotonically from 0.2005 at 20 updates to 0.2794 at
900, while mean native SSL loss falls from 5.1474 to 4.5283. At step 900, every
same-graph diagonal is strongest or close to strongest for its target; the
diagonal accuracies range from 0.2724 (Election) to 0.5270 (Facebook). This
reuses every checkpoint and never inspects downstream labels.

SAMGPT's all-nine native GraphCL trajectory is also complete: 108/108 logical
cells (3 seeds × 4 checkpoints × 9 targets), again with one fixed episode
fingerprint per target. The nine-target mean ROC-AUC is 0.7428 at 20 updates,
0.7239 at 60, 0.7131 at 180, and 0.7127 at 500. The largest 500-minus-20
changes are Election 2020 −0.1869 and COVID Political −0.0644; TwiBot-20
improves +0.0180 and Ukraine Suspended +0.0138. Thus the terminal 500-update
checkpoint is the fixed-compute comparison endpoint, but it is not the
downstream-optimal checkpoint. The full curves must accompany terminal results.

The unified adaptation result is complete. It freezes each
encoder and evaluates budgets 0/1/10/100 examples per class at updates
0/1/10/100 with label seeds 0/1/2, the same deterministic stratified split,
nested labeled samples, a matched 768-to-class linear-head tensor across learned
encoders and raw logistic regression, and ROC-AUC/accuracy/macro-F1 on unchanged
validation and test nodes. Smaller learned representations are zero-padded, so
the full raw 768-dimensional baseline is retained. Zero labels has only update 0
and never constructs or steps an optimizer.
The same grid includes raw logistic regression and an MLP. The validated export
has 3,744/3,744 validation/test rows, 48/48 complete model-target grids, and
retains every training seed, target, label seed, budget, and update cell. A
pre-launch Tucker smoke check also established that `static_train` is not a common view across the
four targets, so all topology-using extractors are registered against each
artifact's canonical `graph.edge_index`; no result was produced with mismatched
or silently missing edge views.

At 100 labels/class and 100 head updates, mean test ROC-AUC across the four
targets and label/training seeds is 0.7836 for PRODIGY, 0.7647 for SAMGPT,
0.7119 for VISION, 0.7112 for raw logistic regression, 0.7011 for raw MLP, and
0.6299 for GraphSAGE. Normalized AUC over log10(labels + 1) is respectively
0.7338, 0.6756, 0.6576, 0.6552, 0.6381, and 0.5729. These are matched-head
results, not each family selecting a different best checkpoint. The complete
optimization curves are preserved; median updates to reach 95% of update-100
ROC-AUC are usually 0–1, except SAMGPT at 100 labels/class (10).

GraphSAGE now also has a complete matched saturation trajectory: 2,184/2,184
validation/test rows over seven exact native link-prediction checkpoints, four
targets, three label seeds, and the full budget/update schedule. At 100
labels/class and 100 head updates, mean ROC-AUC is non-monotonic—0.6252 at
initialization, 0.6087 at step 100, and 0.6299 at step 2,000. The terminal gain
over initialization is only +0.0047, and the target-specific curves differ, so
the result does not support a general monotonic benefit from more pilot-v1
pretraining. This trajectory uses one native training seed.

## Compute-regime boundary

All final-core PRODIGY, VISION, SAMGPT, and GraphSAGE checkpoints credited here
are fixed-compute checkpoints. Existing longer trajectories may describe
saturation, but no result is relabeled as convergence-trained without a declared
stopping rule and provenance. The coverage ledger therefore does not imply a
fixed-compute versus convergence comparison that has not actually been run.

| family | credited terminal checkpoint | registered regime | convergence-trained counterpart |
|---|---:|---|---|
| PRODIGY | 2,500 updates | fixed compute | none registered |
| VISION | 2,500 updates | fixed compute | none registered |
| SAMGPT | 500 updates | fixed compute | none registered |
| GraphSAGE | 2,000 updates | fixed compute | none registered |

The VISION and SAMGPT checkpoint trajectories are saturation diagnostics within
their fixed schedules. They do not retroactively turn the terminal checkpoints
into convergence-selected models.
