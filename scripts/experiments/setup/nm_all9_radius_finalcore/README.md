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

Checkpoint selection uses 500 deterministic validation batches (four episodes per
batch, 2,000 episodes total) on each of three primary panels: radius 2, radius 3, and
global. It selects the largest macro-average across the panels; an exact tie chooses
the earlier checkpoint. Test remains locked until every requested arm/seed selection
exists.

Validation defaults to `VALIDATION_MODE=shared`: each CPU-sampled/collated batch is
forwarded through all four checkpoint models on the same GPU. This preserves the
checkpoint-by-panel cells and their deterministic episode stream while avoiding four
copies of the dominant CPU work. `VALIDATION_MODE=legacy` retains the one-checkpoint-
per-stream implementation for equivalence checks. Test already has only one frozen
checkpoint and therefore uses the legacy single-model stream.

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

For the explicitly reduced one-seed follow-up, use `SEEDS=0` and three GPUs; aggregation
records the requested seed set and reports `seed_std=0` rather than implying a
multi-seed estimate:

```bash
DRY_RUN=1 PHASE=all SEEDS=0 GPUS="0 1 2" \
  bash scripts/experiments/setup/nm_all9_radius_finalcore/run_evaluation_tucker.sh
```

Strict aggregation writes validation trajectories, selected steps, per-seed test rows,
and a seed-aware summary under `log/nm_all9_radius_finalcore_eval/summary/`. Analysis
and findings belong in a matching analysis directory only after these results exist.

## Seed-0 convergence follow-up (10k)

The convergence follow-up reruns all three seed-0 arms from random initialization for
10,000 updates. It writes weights plus an atomic full-state sidecar at 2,500, 5,000,
7,500, and 10,000 completed updates. The sidecar contains the AdamW state, completed
step, Python/NumPy/Torch CPU and CUDA RNG state, and the private episode-sampler state.
Training uses `workers=0` because multiprocessing prefetch can advance worker RNG beyond
the last consumed optimizer step and would make an interruption checkpoint ambiguous.

Evaluation records complete validation trajectories on four matched panels: radius 2,
radius 3, global, and balanced within-source. The within-source panel is diagnostic and
does not influence checkpoint selection; the frozen selected checkpoint is tested on all
four panels.

```bash
DRY_RUN=1 GPUS="0" \
  bash scripts/experiments/setup/nm_all9_radius_finalcore/run_convergence_10k_tucker.sh

tmux new-session -d -s radiusfc10k \
  'export PATH="/home/mhchu/miniconda3/bin:$PATH"; \
   GPUS="0" \
   bash scripts/experiments/setup/nm_all9_radius_finalcore/run_convergence_10k_pipeline_tucker.sh'
```

If interrupted, restart an individual run with the same resolved configuration and
`--resume_training_checkpoint .../checkpoint/training_state_<step>.ckpt`. A historical
`state_dict_<step>.ckpt` remains a weights-only warm start and is deliberately rejected
by the exact-resume option.

## Within-episode distance-stratified follow-up

`distance_stratified.yaml` changes the intervention from one radius per episode to a
mixture inside every episode. An anchor is sampled first. The remaining centers are
drawn from sampled-BFS distance bands and an independent global band outside the sampled
local region. The global band is source-unaware and may cross source graphs.

Allocation is dynamic rather than tied to 30-way training. Positive weights are
converted to integer class counts from the configured `n_way`, with every band receiving
at least one class; the anchor occupies one slot in the first band. Thus radii `2,3` and
weights `1,1,1` resolve to 10/10/10 at 30-way and 5/5/5 at 15-way. Any number of
strictly increasing radii is supported, with one additional weight for the global band.
Because the local region uses bounded fanout sampling, these are sampled-BFS discovery
bands, not claims of exact all-edge shortest-path shells.

Run the dedicated read-only feasibility gate before training:

```bash
bash scripts/experiments/setup/nm_all9_radius_finalcore/run_distance_stratified_preflight_tucker.sh
```

Then dry-run and launch the isolated seed-0 10k pipeline on currently owned GPUs only:

```bash
DRY_RUN=1 GPUS="0 1" \
  bash scripts/experiments/setup/nm_all9_radius_finalcore/run_distance_stratified_10k_pipeline_tucker.sh

tmux new-session -d -s radiusfc_strat10k \
  'export PATH="/home/mhchu/miniconda3/bin:$PATH"; \
   GPUS="0 1" \
   bash scripts/experiments/setup/nm_all9_radius_finalcore/run_distance_stratified_10k_pipeline_tucker.sh'
```

This follow-up has separate state/log roots and does not add a fourth job to the original
three-arm queues. It reuses the same four evaluation panels and checkpoint schedule.
