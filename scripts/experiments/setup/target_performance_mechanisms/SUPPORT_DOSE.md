# Saved-step sensitivity and support-suppression dose

Frozen CPU inference only. Extends the completed role-topology replay without
changing production code, checkpoints, cached queries or target membership.

The full panel uses all six HK/Ukraine controls and every available saved step
(0, 100, 300, 900, 2500), all five targets and both streams. Each step compares
intact inference and support-only background removal. At step 2500, three
shared nested draws add 25/50/75% suppression: 1,140 metric cells in total.
Step counters and weights must agree with the saved exact-training sidecars;
initial and terminal hashes must match the earlier verified arms. Step 0 is
the actual saved initialization, not a newly drawn untrained model.

Mask generation is label-blind. Within each support subgraph, a fixed random
permutation of edge units is shared across doses. A unit is a directed edge,
or a reciprocal pair in fully reciprocal subgraphs; self loops are singleton
units. Round-half-up counts give nested masks, and all background support
edges disappear at 100%. This differs intentionally from the older partial
drop helper that retained self loops. Features, selected nodes, pooling and
all query edges remain fixed. Actual edge fractions and empty cases are saved.

```bash
bash scripts/experiments/setup/target_performance_mechanisms/run_support_dose_tucker.sh --output log/support_dose_dry_20260907 --dry-run
bash scripts/experiments/setup/target_performance_mechanisms/run_support_dose_tucker.sh --output log/support_dose_smoke_20260907 --targets covid_political facebook_page_reference --seeds 0 --steps 0 2500 --max-batches 1
tmux new-session -d -s mechanism-support-dose 'cd /dataMeR1/phil/gfm/prodigy-role-topology && bash scripts/experiments/setup/target_performance_mechanisms/run_support_dose_tucker.sh --output log/support_dose_full_20260907 > log/support_dose_full_20260907.log 2>&1'
```

Do not restart based on an SSH timeout. Inspect the exact tmux session and log
first. Output paths must be new; no overwrite or implicit resume. The job's
worktree must stay at its pinned revision until completion.

The smoke is not research evidence. Full replay requires all 32 batches and
matches the step-2500 intact/removal metrics against the earlier grid. Every
intervened forward checks unchanged query representations before and after
the metagraph, as well as agreement with the original decoder suffix.
No evaluation labels choose a dose, draw, checkpoint or target. Report all
curves, AUC and NLL; masks and streams are not extra training seeds. These
controls test behavior over the saved training window, not beyond step 2500.
