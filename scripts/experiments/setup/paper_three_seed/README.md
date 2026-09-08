# Paper three-seed replication

This queue materializes the two missing PRODIGY training seeds required by the
LoG extended-abstract design. It runs exactly 57 models per seed:

- eight one-hop GraphSAGE specialists;
- 18 distinct multi-source mixtures across the three one-hop source orders;
- eight topical-order two-hop GraphSAGE rungs;
- eight topical-order one-hop GATv2 rungs;
- 15 fixed-exposure two-hop rungs (orders A and C, shared endpoint once).

The queue uses deterministic run names beneath its own experiment worktree and
skips a model only when its 40k comparison checkpoint already exists. The
fixed-exposure configurations retain their registered `10k × source_count`
budgets; every other family is pinned to exactly 40k optimizer steps.

On Tucker, after confirming owned GPUs 0-3 are free:

```bash
DRY_RUN=1 SEEDS="1 2" GPUS="0 1 2 3" bash scripts/experiments/setup/paper_three_seed/run_queue_tucker.sh
tmux new-session -d -s prodigy-paper-seeds \
  'export PATH="/home/mhchu/miniconda3/bin:$PATH"; SEEDS="1 2" GPUS="0 1 2 3" bash scripts/experiments/setup/paper_three_seed/run_queue_tucker.sh > log/paper_three_seed/queue.log 2>&1'
```

For the multi-source conditions, prefer `run_fast_train_tucker.sh`. It uses the
validated shared-graph launcher, defaults to the currently idle GPUs 0, 2, and 3,
six model slots per GPU, and a total loader-worker budget of 72. It runs each
compatible context family as a separate shared-memory batch. The legacy queue
remains the fallback for the eight standalone specialist graphs, the GATv2 ladder
(whose configs load separate source artifacts), and exact historical output naming.

`run_flagship_ladders_tucker.sh` adds seeds 1 and 2 for the five matched
intervention ladders needed by the main figure: baseline, auxiliary objective,
size-proportional exposure, blocked scheduling, and cross-graph episode classes.
It also replicates the eight-rung wide-capacity diagnostic from the presentation.
All 48 conditions per seed share one graph load. The five main curves are the
inferentially matched flagship comparison; capacity remains a secondary mechanism
test and historical ladders remain context rather than being pooled across
protocols.

`finish_core_then_flagship_tucker.sh` is the production orchestrator. It preserves
the already-running seed-1 batch, replaces the reload-heavy remainder with one
shared two-hop load and one shared one-hop load, then trains 96 flagship and
capacity replicas. It follows training with 512 fixed 30-way NM episodes on all nine receiver
graphs and the repository's published 128-episode 2-way/10-shot classification
stream on five labeled targets. The classification phase joins the original seed-0
ladder to seeds 1 and 2, fingerprints every checkpoint and episode stream, and
writes `classification_evaluation/classification_long.tsv` only after validating all
600 cells.

The corrected recovery path uses 14 flagship models per GPU with a total
168-loader-worker budget on GPUs 0, 2, and 3. This matches the live, error-free
concurrency measured when eight remainder trainers and six mechanism trainers
shared each device (about 24 GiB used on an 80 GiB GPU). It reduces the 96-model
replication from four training waves to three while retaining four loader workers
per model; both values remain explicit at the recovery call site.

After training completes, the queue evaluates all checkpoints on the nine fixed
NM receiver episode sets. One-hop GraphSAGE, GATv2, and two-hop GraphSAGE are
evaluated separately so that each checkpoint is reconstructed with the matching
architecture and sampler.

For the optimized shared-run layout, launch `run_fast_core_eval_tucker.sh` in a
separate detached worktree. It waits for the 18 seed-1 one-hop, 18 seed-2
one-hop, and 46 two-hop/fixed-exposure jobs plus the flagship queue's GPU release.
It then reconstructs each terminal model from its recorded effective config and
evaluates one fixed 512-episode panel from the registered all-nine final-core
graph in a single shared graph load. The separate evaluation graph is required
because the core models trained on all8 while the paper panel includes Facebook.
The audit
requires exactly 82 models x 9 targets = 738 cells, one episode fingerprint per
target, unique family-qualified model IDs, finite metrics, exact checkpoint
steps, checkpoint hashes, and training revisions. This adapter is required
because the optimized runs deliberately do not use the legacy deterministic
`state/paper3seed_*` paths.

Launch `postprocess_flagship_tucker.sh` from a separate, current-revision
worktree before the training queue finishes. It waits for the exact 864-cell
seed-1/2 NM grid and 600-cell three-seed classification table, checks their
cardinalities and basic metric contracts, and then runs the preregistered
whole-ladder/endpoint decision plus the two-panel seed-band figure. This keeps
postprocessing code current without changing a live training or evaluation
worktree.
