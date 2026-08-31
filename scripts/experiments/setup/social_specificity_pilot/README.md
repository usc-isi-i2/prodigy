# One-seed social-specificity pilot

This pilot asks whether PRODIGY's neighbor-matching representation is specific to
social-media graphs or transfers similarly across graph domains. It builds the smallest
directional matrix with two social sources/targets and two citation sources/targets:

- Ukraine/Russia Twitter;
- Facebook Page Reference;
- Cora;
- PubMed.

The 4×4 matrix has 16 cells. The existing seed-0, step-2,500 final-core Twitter and
Facebook specialists are reused. Only the Cora and PubMed specialists are trained.
Both use the final-core architecture and fixed-compute protocol: two hops, fanouts
`9,9`, 101-node cap, 30-way/3-shot/4-query neighbor matching, batch size 4, and 2,500
updates. Evaluation uses the cataloged static train/test edge views and the repository's
fixed split-derived episode stream.

The original Cora and PubMed artifacts contain the older 85/15 static split. The
launcher leaves those canonical files unchanged and creates resumable pilot-local
70/15/15 copies beneath `state/social_specificity_pilot/data/`. Social graph files are
linked read-only into the same pilot-local data layout so the shared evaluator can use
one data root.

This is a directional screen, not a training-seed uncertainty study. The eventual
analysis may bootstrap paired evaluation episodes and compare the effect scale with
the existing three-seed final-core variation, but it must not call either quantity the
training-seed variance of the new citation models.

## Validate and dry-run

Run these on Tucker from a dedicated worktree after confirming no job is already using
it:

```bash
python scripts/experiments/setup/social_specificity_pilot/validate_plan.py --check-data

DRY_RUN=1 GPUS=2,3 \
  bash scripts/experiments/setup/social_specificity_pilot/run_tucker.sh
```

The launcher refuses GPUs other than 2 and 3. It writes checkpoints and logs only under
the invoking worktree's `state/social_specificity_pilot/` and
`log/social_specificity_pilot/` unless explicitly overridden. It refuses to continue
from an incomplete same-stamp training directory and never overwrites a checkpoint.

## Launch

The user normally starts long Tucker jobs. From the dedicated worktree:

```bash
tmux new-session -d -s socialspec-pilot \
  'export PATH="/home/mhchu/miniconda3/bin:$PATH"; \
   GPUS=2,3 bash scripts/experiments/setup/social_specificity_pilot/run_tucker.sh \
   > log/social_specificity_pilot/orchestrator.log 2>&1'
```

Inspect progress with:

```bash
tail -f log/social_specificity_pilot/orchestrator.log
```

After results exist, create the name-aligned analysis folder and report the four block
means: social→social, social→citation, citation→social, and citation→citation. Normalize
each transfer cell by its target's in-domain specialist value. Treat a cross-domain
deficit below 0.02 as a likely generic-transfer result, above 0.05 in both directions
as a seed-expansion candidate, and an intermediate or asymmetric result as requiring
seeds 1 and 2 before interpretation.
