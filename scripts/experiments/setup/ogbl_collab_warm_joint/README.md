# Warm-positive joint training: frozen six-cell test

Hypothesis: allocating positive updates to the historical warm-endpoint population
improves novel-collaboration recall in the official warm-endpoint validation cohort.
This tests population matching, not a new architecture or an AA-preservation rule.
An older 481-parameter gate benefited modestly from warm filtering; the original
joint campaign used warm positives but a different training year and selection.
Neither is this matched comparison against the current fresh-negative joint model.

Control and treatment: seeds 0/1/2, 2000 updates, evaluate every 50, earliest maximum
standalone 2018 Hits@50. Control uses all 119622 historical 2017 positives;
treatment uses 47546 with both endpoints incident to the strictly pre-2017 graph.
2048 positives with replacement per update, 1024 uniform and 1024 mined negatives,
fixed fresh 100000-negative panel, top2048 mining refreshed every10, balanced BCE,
Adam .001, weight decay .0001, gradient clip5. Positive identities and repeated
exposure change; model-dependent hard-negative identities can also change. This
is not equal epochs. Same seeds and budgets do not imply identical sample streams.
All graph edges, negative membership, standardization, static author inputs,
architecture and validation data remain unchanged. 496385 neural parameters;
496448 conservative inference scalars. The eligibility rule adds no fitted scalar.

Graph-derived eligibility must equal the prior historical warm cohort, with
exact pair-set equality. All source files are checked against existing manifests.
Controls must reproduce the archived 120 validation measurements. Selected
checkpoints must replay exactly and agree with the official OGB evaluator.
Record full curves, cohorts, hashes, and offline W&B for all six cells.
Report full-panel net recoveries/losses and novel/repeat, zero/nonzero-AA slices.
Training-panel and same-year-probe metrics use training positives (the probe keeps
all positives); these are diagnostics and not independent generalization scores.

Advance only if every selected paired difference is positive and mean gain >=.005.
Otherwise stop; no grid expansion. A pass warrants separate historical replication,
not automatic test access. No 2019 reads, refit, test scoring or submission.
2018 has been repeatedly explored, including the motivation for this intervention.
Prior 2019 exploration and invalid negative-leakage/test-supervised campaigns remain
disclosed. Supplied static features are not historical snapshots. This is adaptive
research and does not establish leaderboard acceptance.

Dedicated Tucker worktree prodigy-collab-warm-joint; runtime warm_joint_v1 under
ogbl_collab_compact_joint. GPU1 only, six sequential cells. Prior matched training
runs suggest several minutes of training plus environment startup and audit.
