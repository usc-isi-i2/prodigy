# All-nine center-radius NM on the final-core protocol

This experiment tests whether controlling the graph-distance scale of competing
neighbor-matching classes teaches PRODIGY to discriminate both locally similar and
globally distant nodes. It is isolated from `setup/final_core/`: it has its own configs,
state/log roots, feasibility gate, launchers, checkpoint selection, and result schema.

## Controlled intervention

`n_hop` remains the final-core fair two-hop context sampler (`9,9`, 101 nodes), and
one-hop random walks still define the seven support/query positives for every class.
The new parameter changes only where the 30 class centers in an episode come from.

For a finite radius, the sampler chooses a node-uniform anchor and builds a bounded,
fanout-sampled ball on the leakage-free `static_train` adjacency. Every selected center
is therefore at graph distance at most the configured radius from that anchor. For
`global`, centers are sampled from all eligible nodes in the complete all-nine merge.

The merge is a disjoint block concatenation. Consequently:

- finite radius 2/3 episodes remain in one connected source component naturally;
- global episodes can contain centers from several source graphs; and
- no arm uses `graph_id` source confinement, strata, source subsets, or source balancing.

The radius is selected independently for each of the four episodes in an optimizer
batch. Center and support/query nodes are globally unique within an episode: a node can
never receive two class labels, and a class center cannot also become a target example.

## Arms

| arm | per-episode center radii | purpose |
|---|---|---|
| `global` | global | source-unaware causal control |
| `radius_mix` | 2, 3, global with equal weights | close-and-far treatment |
| `close_only` | 2, 3 with equal weights | all-close third arm |

Each arm uses seeds 0, 1, and 2: nine training runs total.

## Final-core contract

All other settings are copied from the 2026-08-07 final-core experiment:

- immutable all-nine seed-0 artifact with 70/15/15 unordered-edge split;
- `static_train` positives/context during training and disjoint validation/test positives;
- `256 · S,U,M`, one-layer GraphSAGE, no dropout;
- 30-way, 3-shot, 4-query, batch size 4;
- learning rate 0.002 and weight decay 0.001;
- 2,500 optimizer updates; and
- checkpoints after 100, 300, 900, and 2,500 completed updates.

This is not a 40k-step ladder experiment.

## Mandatory feasibility gate

Run the read-only sampler probe before any GPU job. It loads the immutable graph and
constructs 100 full 30-way episodes at radius 2, radius 3, and global on each of the
training, validation-positive, and test-positive views. It fails unless all episodes
are collision-free, every finite episode stays in one source component, and global
sampling actually demonstrates cross-source coverage.

```bash
cd /dataMeR1/phil/gfm/prodigy-radiusfc
bash scripts/experiments/setup/nm_all9_radius_finalcore/run_preflight_tucker.sh
```

The gate writes
`log/nm_all9_radius_finalcore/preflight/feasibility.json`. Training refuses to start
without a report whose top-level `ready` value is true.

## Tucker isolation

Create a dedicated worktree only after the currently important jobs finish. Check
`tmux ls`, `git worktree list`, GPUs 0-3, and host memory first. Never pull or switch a
worktree that owns a live process.

```bash
cd /dataMeR1/phil/gfm
git -C prodigy fetch origin codex/nm-all9-radius-finalcore
git -C prodigy worktree add ../prodigy-radiusfc \
  origin/codex/nm-all9-radius-finalcore
cd /dataMeR1/phil/gfm/prodigy-radiusfc
git config core.hooksPath .githooks
```

## Dry run and smoke

The launcher only accepts owned GPUs 0-3, refuses busy GPUs, checks host RAM, refuses
ambiguous partial state directories, and defaults to one graph-loading process per GPU.

```bash
DRY_RUN=1 MODE=smoke GPUS="0" \
  bash scripts/experiments/setup/nm_all9_radius_finalcore/run_training_tucker.sh

tmux new-session -d -s radiusfc_smoke \
  'export PATH="/home/mhchu/miniconda3/bin:$PATH"; \
   MODE=smoke GPUS="0" \
   bash scripts/experiments/setup/nm_all9_radius_finalcore/run_training_tucker.sh'
```

Require a finite loss and an honest step-5 checkpoint for all three arms before the
full queue.

When Tucker is intentionally shared with a low-utilization job, the busy-GPU check can
be overridden explicitly with `ALLOW_BUSY_GPUS=1`. The ownership and host-RAM checks
remain mandatory. Use one GPU and one slot, and apply low CPU/I/O priority to the outer
launcher; never combine this override with parallel graph loads.

## Training

```bash
DRY_RUN=1 MODE=train GPUS="0 1 2 3" \
  bash scripts/experiments/setup/nm_all9_radius_finalcore/run_training_tucker.sh

tmux new-session -d -s radiusfc_train \
  'export PATH="/home/mhchu/miniconda3/bin:$PATH"; \
   MODE=train GPUS="0 1 2 3" \
   bash scripts/experiments/setup/nm_all9_radius_finalcore/run_training_tucker.sh'
```

## Frozen evaluation

Checkpoint selection uses 500 fixed validation episodes on each of three primary
panels: radius 2, radius 3, and global. It selects the largest macro-average across the
panels; an exact tie chooses the earlier checkpoint. Test remains locked until all nine
selections exist.

The frozen checkpoint is tested once on the same three panels plus the historical
balanced within-source panel as a secondary compatibility diagnostic. The compatibility
panel is never used for checkpoint selection.

```bash
DRY_RUN=1 PHASE=all GPUS="0 1 2 3" \
  bash scripts/experiments/setup/nm_all9_radius_finalcore/run_evaluation_tucker.sh

tmux new-session -d -s radiusfc_eval \
  'export PATH="/home/mhchu/miniconda3/bin:$PATH"; \
   PHASE=all GPUS="0 1 2 3" \
   bash scripts/experiments/setup/nm_all9_radius_finalcore/run_evaluation_tucker.sh'
```

Strict aggregation writes validation trajectories, selected steps, per-seed test rows,
and a three-seed summary under `log/nm_all9_radius_finalcore_eval/summary/`. Analysis and
findings belong in a matching analysis directory only after these results exist.
