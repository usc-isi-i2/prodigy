---
name: prodigy-controlled-campaigns
description: Design or extend multi-arm experiment campaigns in this repository, including ladders, matrices, ablation grids, leave-one-out studies, multi-seed comparisons, checkpoint trajectories, and staged intervention selection. Use when validity depends on freezing a matched protocol and tracking many resumable cells across any model family.
---

# Controlled Experiment Campaigns

Turn a research question into a frozen, enumerable, resumable campaign whose cells
remain comparable. Use this for multi-arm work where scheduling, selection, missing
cells, or reused artifacts could otherwise change the estimand after results appear.

The `prodigy-` name is retained for repository-skill consistency. Campaigns may span
PRODIGY, SAMGPT, GraphSAGE, VISION, GILT, or future model families.

## Before acting

1. Use `prodigy-experiments` for repository layout and the closest existing setup and
   analysis pattern.
2. Read [experiment workflow](../../../docs/agent_guides/experiment_workflows.md) and
   [repository traps](../../../docs/agent_guides/repo_traps.md).
3. Inspect a comparable campaign's manifest, planner, validators, tests, and findings;
   do not copy only its launcher.
4. Use `prodigy-tucker` for cluster execution and
   `prodigy-evaluation-integrity` before admitting results as evidence.

## Freeze the campaign contract

Before inspecting held-out outcomes, record:

- research question, primary estimand, baseline, and minimum required controls;
- graph/source/task/rung/seed grid and the semantic identity of every cell;
- fixed splits, episode or sample policy, budgets, schedules, and checkpoints;
- stopping, selection, compatibility, exclusion, and tie-breaking rules;
- smoke, training, validation/selection, final evaluation, and analysis phases;
- expected outputs, completeness criteria, and treatment of failed or missing cells;
- authorized compute and any action the user is expected to launch.

Do not call a comparison matched unless its differing and controlled variables are
enumerated. If selection uses validation outcomes, keep final held-out outcomes closed
until the selection artifact is frozen.

## Workflow

1. Generate a machine-readable manifest or plan from semantic fields. Give each run
   and evaluation cell a stable, collision-resistant identity.
2. Validate the grid before launch: duplicates, omissions, incompatible arms, late-
   failing task names, unavailable graphs, mismatched budgets, and output collisions.
3. Provide a dry run that reports exact commands or jobs, resource allocation,
   expected checkpoints, output locations, and total campaign size.
4. Separate phases so smoke artifacts cannot be mistaken for production and test
   outcomes cannot influence training, stopping, or selection.
5. Make resume decisions from validated checkpoint/config/provenance identity, not
   filenames or directory existence alone. Retain completed valid cells and enumerate
   only genuinely pending work.
6. Preserve paired inputs or fingerprints when the estimand requires matched episodes,
   examples, schedules, or sampling streams. Verify the claimed match directly.
7. Validate completion against the frozen grid and emit explicit missing, failed,
   rejected, and reused cell inventories.
8. Aggregate according to the predeclared estimand and missingness rules. Record
   operational deviations and manual interventions append-only.

## Required output

Keep reproduction instructions and the frozen campaign contract with setup. Keep
admitted result data, figures, validation receipts, and findings in the matching
analysis leaf. Report campaign completeness and protocol deviations before reporting
the headline result.

## Maintenance

Prefer shared planners, manifests, validators, and harnesses once two or more campaigns
need the same behavior. Update this skill when campaign structure changes across the
repository; keep one-off scientific choices in the experiment's frozen protocol.
