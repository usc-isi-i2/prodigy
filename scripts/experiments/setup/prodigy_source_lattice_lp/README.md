# PRODIGY source-lattice static LP

Evaluates the seed-0 final-core lattice (9 specialists, 36 unordered pairs, and
9 leave-one-out models) on all nine social targets with the repaired frozen
pair-link protocol. The sweep uses one shared, deterministic degree-matched pair
set per target, validation-locked cosine orientation, background-only message
passing, 2,000 held-out positive edges, and explicit leakage/endpoint gates.
The background/holdout partitions are loaded from the canonical seed-0 caches
shared with the GraphSAGE and SAMGPT evaluations, including for graph artifacts
that do not embed named static-link views.

Run on Tucker from this worktree with GPUs 0-3:

```bash
tmux new-session -d -s prodigy-lp-all9 \
  'export PATH="/home/mhchu/miniconda3/bin:$PATH"; bash scripts/experiments/setup/prodigy_source_lattice_lp/run_tucker.sh'
```

The launcher is resumable and refuses to report completion unless all 486 model
by target cells are present. Outputs remain under
`log/prodigy_source_lattice_lp/`; checkpoints are read-only inputs from their
original isolated training worktrees.
