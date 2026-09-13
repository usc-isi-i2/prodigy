---
name: prodigy-evaluation-integrity
description: Validate, import, merge, compare, repair, or cite evaluation evidence in this repository, including logs, checkpoints, metric files, replay outputs, and shared result tables. Use before treating experimental output as research evidence; applies across PRODIGY and comparison models.
---

# Evaluation Integrity

Decide whether an artifact is admissible evidence before interpreting it. A
successful job or plausible metric is not sufficient: this repository contains
historical and parser failure modes that can silently produce incomplete or invalid
comparisons.

The `prodigy-` name is retained for repository-skill consistency. Apply this workflow
to every model family evaluated here, not only PRODIGY.

## Before acting

1. Read [repository and evaluation traps](../../../docs/agent_guides/repo_traps.md).
2. Read the producing experiment's setup README, analysis findings, and relevant
   evaluator or parser. Do not reconstruct the protocol from filenames alone.
3. If artifacts exist only on Tucker, also use the `prodigy-tucker` skill.

## Evidence contract

Establish, when available:

- producing revision, resolved config, run identity, and model family;
- checkpoint path, numeric step, and intended comparison step;
- graph, task, split, shot count, ablation, seed semantics, and evaluator;
- metric definition, fixed-panel or replay identity, and expected output cells;
- destination evidence table and its deduplication key.

Mark unknown fields as unknown. Do not infer protocol identity from similar run names.

## Workflow

1. Apply the historical validity gates in `repo_traps.md`, especially for link
   prediction, pre-fix terminal checkpoints, feature ablations, and evaluation seeds.
2. Verify task and intervention identity survives parsing and is represented in the
   deduplication key. Inspect raw metric JSON when the shared parser cannot represent
   the run faithfully.
3. Pin comparison checkpoints explicitly. Do not compare each arm's highest numbered
   checkpoint unless the protocol establishes that those steps are equivalent.
4. Verify checkpoint hashes, resolved configs, manifests, episode fingerprints, or
   replay parity when the experiment records them. Use the experiment's declared
   tolerances; do not weaken them merely to admit a result.
5. Reconcile expected versus observed cells. Missing required cells mean incomplete,
   not negative or zero evidence.
6. Preserve accumulated result tables as keyed set unions. Before committing a merge,
   prove that required rows from every input or parent remain present.
7. Keep committed CSV/JSON evidence under designated `data/` paths and figures under
   `figures/`; verify ignore and tracked-file status after moves.
8. Classify artifacts as valid evidence, smoke-only, incomplete, historically void,
   or protocol-incomparable. Carry that classification into findings and summaries.

## Required output

Report the evidence contract, validation checks, missing or rejected artifacts, and
the claims the admitted evidence actually supports. Preserve a machine-readable
validation receipt when the experiment already uses receipts or when the validation
is too detailed to audit reliably from prose alone.

## Maintenance

Treat `repo_traps.md`, shared evaluators, and parser tests as the current authority.
When a new demonstrated failure mode changes evidence admission across experiments,
update that guide and this skill together; avoid accumulating rules for isolated
mistakes that existing tests or experiment-local documentation already contain.
