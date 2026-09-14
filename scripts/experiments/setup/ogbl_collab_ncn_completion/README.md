# Compact NCN / one-step completion: frozen validation-only campaign

Question: does learned pair-specific neighborhood pooling improve the current fresh
joint scorer, and does soft missing-neighbor completion improve observed pooling?
Nine cells: control / observed / completion × seeds 0,1,2. All cells complete 2000
updates with validation every50; select the earliest maximum standalone 2018
Hits@50. Use the admitted 2017 fresh-negative training panel and pre-2017 graph;
2018 validation uses its admitted pre-2018 graph. No refit or 2019 reads/scoring.

Same 256-wide, three-layer SAGE encoder and exactly paired encoder initialization.
Same 2048 positive/1024 uniform-negative/1024 mined-negative sampling slot streams,
fixed100k negative pool, top2048 mining refreshed every10, BCE, Adam .001, weight
decay .0001, clip5. Positive identities and uniform/mining slots match; actual
hard-negative identities depend on model scores. All historical positives retained.
This is equal updates, not equal compute. No new tail loss or renewal treatment.

Control is the unchanged 496385-parameter joint model with symmetric14 handcrafted
features and endpoint product/absolute difference. New heads replace handcrafted14
with pooled learned common-neighbor representations: shared projection256->64 ReLU,
product/absolute difference512 + pool64 ->256->128->1 ReLU MLP. Each new model has
525633 neural parameters. Conservative inference scalar budgets496448/525697/525697
include the original63 auxiliary scalars even though new heads do not use the old
feature standardization, plus one fixed completion prior. No learned node-ID table
or second pretrained model. This is a scoring-family comparison, not a capacity-only
or one-feature ablation. Observed versus completion has identical architecture and
initial weights and isolates the completion computation.

Actual common neighbors are pooled with a cap32. Candidates for completion come
from each endpoint's neighbors absent from the other endpoint's neighborhood,
excluding both target endpoints; cap8 each side. Sampling uses a fixed node priority
permutation (seed314159), independent of target labels, and expansion by the full
eligible-count / selected-count. Padding is a zero vector, never a real node.
Candidate missing links are deduplicated and scored by this model's observed decoder
(one recursion level only). Their detached soft weights are sigmoid(logit+log(.1)).
Training weights refresh every10 updates with mining; validation and selected replay
always compute fresh weights. No predicted edges are inserted into message-passing
adjacency. Caches store graph-derived arrays, not fitted embeddings. Report memory
and inference cost separately from the learned-parameter budget.

This is NCNC-inspired, not a reproduction of the paper. Differences include the
retained SAGE encoder, compact projection and decoder, no dropout, identical fixed
candidate subsampling at train/inference, a fixed prior without a tuning grid, and
training completion weights cached for up to9 updates. The paper's public source
uses shared observed prediction for one-step completion with detached weights:
https://github.com/GraphPKU/NeuralCommonNeighbor/blob/main/model.py
Paper: https://proceedings.iclr.cc/paper_files/paper/2024/hash/3efb4bdc6bfe13e1ff95b4407c37961d-Abstract-Conference.html

Advance each new arm only if all3 paired gains over control are positive and mean
>=.005. If both pass, higher mean wins, observed wins exact tie (less computation).
A pass supports separately frozen earlier-year replication, not automatic test
access. Otherwise stop; no grid expansion. All40 sampled validation misses are
illustrative only, not a fitted training subset. All2018 development and prior2019
exploration remain disclosed, including prior negative leakage and test-supervised
experiments. No leaderboard acceptance claim. Static node features are not historical
snapshots. Three seeds represent optimization variation on the same data.

Before production: synthetic tests for graph membership, endpoint symmetry,
self/padding exclusion, sampling-mass expansion, zero-completion equality, finite
nonzero gradients, and paired encoder initialization; training-only ten-update
smoke and capacity profile per arm. Freeze source and manifest before production.
Control must reproduce all120 archived validation values. Verify all9 checkpoint
replays, official OGB metric parity, independent sorted-cutoff metrics, source/hash
identities, RNG slot fingerprints, completeness and full-panel error strata.

Run `launch.sh prepare` for tests, cache construction and three training-only smoke
profiles. Run `launch.sh production` for the nine cells after inspecting profiles.
Dedicated Tucker worktree prodigy-collab-ncn-completion; runtime under
ogbl_collab_compact_joint/ncn_completion_v1. GPU1 unless capacity profiling and live
ownership checks justify another owned GPU. Offline W&B for substantive runs.
