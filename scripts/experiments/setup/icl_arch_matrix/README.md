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
