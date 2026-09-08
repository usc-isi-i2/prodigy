# Matched paper mechanism sweeps

This compact campaign closes three gaps from the paper presentation while reusing one
all-nine graph load. It trains five balanced-source episode-mixture ratios
(`p = 0, .1, .25, .5, 1`) and a 512-dimensional `p=0` encoder, at seeds 0/1/2.
Every run lasts exactly 10,000 updates and preserves checkpoints at 2k increments.

The ratio sampler changes only whether an episode spans source graphs. Both branches
retain balanced source probability, and TwiBot-20 remains excluded from training and
selection. This avoids the size-proportional exposure confound in the older one-seed,
two-target ratio study.

The fixed checkpoint grid separates data volume from source count. The standard and
wide `p=0` trajectories add a capacity-by-budget diagnostic on all nine NM receivers.
The five standard-width ratio trajectories also use the fixed 128-episode,
2-way/10-shot downstream classification stream on five targets. The wide arm is NM-only
because the current external classification evaluator reconstructs one fixed encoder
width; it must not be presented as downstream capacity evidence.

Exact expected coverage is 18 training jobs, 90 physical checkpoint models, 810 NM
cells, and 375 classification cells. Training seeds are the replication unit. Fixed
episode fingerprints make arm comparisons paired but do not create evaluation-seed
replicates.

On Tucker, from a dedicated worktree and with GPUs 0-3 idle, the whole campaign
can still be run in one process:

```bash
tmux new-session -d -s paper-mechanism-sweeps \
  'export PATH="/home/mhchu/miniconda3/bin:$PATH"; bash scripts/experiments/setup/paper_mechanism_sweeps/run_tucker.sh'
```

`run_tucker.sh` also accepts `PHASE=train` and `PHASE=eval`.  The production
`wait_and_run_tucker.sh` uses that split to train the 18 physical models on GPU 1
as soon as the VISION queue releases it, concurrently with flagship work on GPUs
0, 2, and 3.  It then waits for the flagship and core audits, requires two minutes
of stable idle state on all four owned GPUs, and resumes the exact NM and
classification evaluations.  The evaluation phase validates the complete
18-job ledger and all 90 checkpoints before scoring.  Neither phase touches
GPUs 4--7.
