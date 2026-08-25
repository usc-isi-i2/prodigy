# Native-model result-matrix audit

Last updated: 2026-08-25.

## Coverage snapshot

| model/version | SSL→CLS saturation | cross-SSL matrix | downstream CLS matrix | mixture diversity→CLS | adaptation efficiency |
|---|---|---|---|---|---|
| PRODIGY final-core 2.5k | complete | complete | complete | complete | runner ready; results pending |
| VISION native feature similarity | complete | missing | complete | missing | runner ready; results pending |
| GILT native SSL | missing | missing | missing | missing | missing |
| SAMGPT native GraphCL | complete | complete | complete | partial | runner ready; results pending |
| GraphSAGE pilot-v1 | missing | missing | narrow TwiBot probe only | missing | runner ready; results pending |
| MLP | N/A | N/A | supervised reference complete | N/A | matched-budget results pending |
| raw logistic regression | N/A | N/A | raw-feature reference complete | N/A | matched-budget results pending |

This table distinguishes a complete registered design from a broad but
unregistered pile of files. In particular, SAMGPT downstream CLS is now complete
for its registered final-core design: 31 physical source-mixture models by nine
targets by three training seeds. VISION has valid native single-source and
all-nine evidence, but not yet a registered native cross-SSL or mixture-diversity
design.

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
- GraphSAGE pilot-v1 has a valid all-graph checkpoint after 2,000 native
  link-prediction updates. The existing TwiBot full-train node probe is retained
  as a narrow downstream reference, not treated as the requested adaptation
  grid.
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

SAMGPT's all-nine native GraphCL trajectory is also complete: 108/108 logical
cells (3 seeds × 4 checkpoints × 9 targets), again with one fixed episode
fingerprint per target. The nine-target mean ROC-AUC is 0.7428 at 20 updates,
0.7239 at 60, 0.7131 at 180, and 0.7127 at 500. The largest 500-minus-20
changes are Election 2020 −0.1869 and COVID Political −0.0644; TwiBot-20
improves +0.0180 and Ukraine Suspended +0.0138. Thus the terminal 500-update
checkpoint is the fixed-compute comparison endpoint, but it is not the
downstream-optimal checkpoint. The full curves must accompany terminal results.

The unified adaptation implementation is committed locally. It freezes each
encoder and evaluates budgets 0/1/10/100 examples per class at updates
0/1/10/100 with label seeds 0/1/2, the same deterministic stratified split,
nested labeled samples, matched 256-dimensional head initialization across
learned encoders, and ROC-AUC/accuracy/macro-F1 on unchanged validation and test
nodes. Zero labels has only update 0 and never constructs or steps an optimizer.
The same grid includes raw logistic regression and an MLP. A pre-launch Tucker
smoke check also established that `static_train` is not a common view across the
four targets, so all topology-using extractors are registered against each
artifact's canonical `graph.edge_index`; no result was produced with mismatched
or silently missing edge views.

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
