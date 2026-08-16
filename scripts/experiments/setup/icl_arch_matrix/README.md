# PRODIGY × VISION × GILT architecture matrix

This is a lightweight, descriptive comparison—not a final paper-number run.

- Seed: `0` only.
- Pretraining: the final-core 31 physical source sets, all using the same source-confined
  30-way/3-shot/4-query neighbor-matching stream.
- Budget: 100 optimizer updates, batch size 4 (400 episodes per model), with checkpoints
  at 20, 60, and 100 updates.
- Architectures: repository PRODIGY and thin adapters over pinned official VISION/GILT
  implementations (see `upstream_pins.json`).
- GILT keeps its official model/optimizer defaults but uses one frozen seed-0 orthogonal
  768-to-128 feature alignment. This prevents fitting a separate PCA on each downstream
  target (and prevents source-set leakage through a PCA fit over excluded source graphs).
- Downstream task: tuning-free binary node classification for all three architectures.
- Evaluation: 10-shot, 128 fixed seed-0 episodes on `covid_political`, `election2020`,
  `ukr_rus_suspended`, and `twibot20`; query counts follow the graph catalog (12, 1, 1,
  and 12 per class). ROC-AUC is primary; accuracy and F1 are retained as diagnostics.
- Every result records a hash of the actual support/query nodes. Aggregation refuses the
  grid unless the hashes agree across all architectures. The episode RNG is reset after
  architecture-specific model initialization, and the audit includes global centers,
  sampled neighborhood node IDs, and sampled edges.
- VISION/GILT adapter updates backpropagate the four episode losses sequentially before
  one optimizer step. This preserves the registered effective batch of four while avoiding
  retention of four large architecture graphs in GPU memory.

`run_checkpoint_trajectory_tucker.sh` evaluates the four labeled single-source models on
all four classification targets at checkpoints 20 and 60 by default. Together with the
registered update-100 results, this yields both 12 self-graph trajectories and checkpointed
off-diagonal transfer cells. `CHECKPOINT_STEPS` can select any subset of `20 60 100`.
`evaluate_prodigy.py --include-facebook --datasets facebook_page_reference` extends a
PRODIGY evaluation with the Facebook page-reference graph's primary 30-way page-category
labels without changing the frozen four-target matrix protocol.

Use `MODEL_IDS=ss_ukr_rus` with the Tucker launchers for a one-cell pilot. A full launch
uses all 31 source sets. Only Tucker GPUs 0 and 1 are accepted by the training launcher.
`run_matrix_tucker.sh` waits for both owned GPUs, runs and hash-validates that pilot, and
only then advances to the remaining matrix and full aggregation.

If an adapter OOM leaves a partial run, preserve its state and log under a timestamped
archive, create a recovery worktree at the fixed commit, and launch
`queue_oom_recovery_tucker.sh`. It waits for the original tmux session and both owned GPUs,
then resumes only missing terminal checkpoints into a separate recovery log root.
For the recorded worker-0 OOM, `queue_worker0_recovery_tucker.sh` can recover its exact
remaining parity-assigned jobs on GPU 0 while the complementary GPU-1 worker finishes;
each recovered job writes to a distinct log directory and the final queued recovery still
performs the complete-grid evaluation gate.
If GPU 0 becomes occupied by an unrelated user, archive any partial retry and use
`queue_gpu1_final_recovery_tucker.sh`: it waits for the original GPU-1 worker, trains all
remaining checkpoints only on GPU 1, and delays the two-GPU evaluation until both devices
are below the free-memory threshold.

## Random-initialization control

`run_random_init_tucker.sh` evaluates one deterministic untrained instance of each
architecture on the same four fingerprinted episode streams.  It does not associate the
untrained model with any source set.  The aggregator requires exactly 12 cells, step 0,
seed 0, no sources, cross-architecture fingerprint agreement, and exact agreement with
the frozen 372-cell trained matrix fingerprints.  It reports random-init ROC-AUC, the
update-100 mean, their delta, and the fraction of update-100 source-set models above the
architecture/target-specific random baseline.  Set `DEVICE=0` or `DEVICE=1`; any other
GPU is rejected and a busy owned GPU causes a no-clobber exit.

## Raw-feature-only controls

`run_raw_features_tucker.sh` evaluates two topology-free controls on the exact same
four fingerprinted episode streams: cosine nearest-prototype classification and a
fixed L2 logistic regression (`C=1`, `liblinear`). Both use only each center node's
L2-normalized raw 768-dimensional text feature. The prototype is computed from the
20 support nodes in each episode; logistic regression is also fit only on those
support nodes. There is no pretraining, query-label tuning, neighborhood aggregation,
or target-wide fitting. The aggregator requires exactly eight rows and rejects any
episode fingerprint that differs from the frozen trained matrix. This launcher is
CPU-only; it does not reserve or inspect a Tucker GPU.

## Topology-only controls

`run_topology_features_tucker.sh` evaluates two label-free structural controls on
the same four fingerprinted episode streams. Each target graph is represented by
three directed degree features: target-graph-z-scored `log1p` in-degree,
out-degree, and total degree. Cosine prototypes and fixed L2 logistic regression
are fit only on the 20 support nodes in each episode. The controls use the loaded
target graph's edge index but never node text, query labels, or target-wide label
fitting. This is a transductive degree/topology floor, not a trained GNN or label-
propagation result. The CPU-only launcher rejects any episode fingerprint that
differs from the frozen trained matrix.

## Target-supervised references

`run_supervised_target_tucker.sh` trains one target-specific raw-feature MLP and
one two-layer GraphSAGE per classification target. Both use the repository's
deterministic 60/20/20 stratified node split, seed 0, and 100 updates. A fixed
two-value learning-rate grid (`1e-3`, `3e-4`) is selected by 32 validation
episodes at update 100. Final scoring uses only query nodes from the exact 128
frozen test episodes; test support labels and query labels are not used for
fitting or selection. These models have much more target supervision than the
10-shot in-context systems and are therefore supervised references, not matched-
label-budget competitors. The launcher uses only Tucker GPUs 0 and 1 and refuses
to start on an occupied device. Set both `GPU_MLP` and `GPU_GNN` to the same owned
GPU to run the models serially when only one device is available.
