---
name: prodigy-paper-program
description: Plan, prioritize, or interpret PRODIGY research in relation to a candidate manuscript, paper claim, figure, reviewer risk, or submission decision. Use before proposing substantial new experiments and when deciding what research or writing should happen next; do not assume any one manuscript is the active paper.
---

# PRODIGY Paper Program

Make the paper program the parent of experiments. Optimize for a coherent,
defensible manuscript, not for the number of completed runs.

## Establish the active paper

Paper work lives in the sibling repository `/Users/philipp/projects/gfm/paper`.
It contains multiple manuscripts and historical directions. Never infer that
`transfer-prediction/` or the most recently edited manuscript is automatically the
active paper.

Before prioritizing consequential work:

1. Inventory current manuscript candidates, their claim ledgers, submission plans,
   and recent Git history in the paper repository.
2. Identify the active manuscript from an explicit user choice or unambiguous current
   planning evidence. Distinguish active, candidate, paused, superseded, and submitted
   manuscripts.
3. If multiple candidates remain plausible and the choice changes what should be done,
   present the conflict and ask the user to choose. Continue with work that is useful
   across candidates when possible.
4. State the active manuscript and current paper-level objective when reporting a
   prioritization decision.

Do not silently create a new paper direction. Do not treat a planning note, experiment
folder, or attractive result as a manuscript commitment.

## Build the paper spine

For the active manuscript, maintain or reconstruct a compact program view:

- one-sentence thesis and intended contribution;
- the smallest set of central claims needed to support it;
- exact figures, tables, analyses, and citations supporting each claim;
- evidence status: supported, provisional, contradicted, missing, or invalidated;
- the strongest known alternative explanation or reviewer objection;
- the next decision needed, its owner artifact, and a stopping condition;
- manuscript section where the resolved evidence will be integrated.

Prefer an existing claim ledger or submission plan. Do not create another overlapping
planning document merely because the existing one is imperfect. When a durable program
artifact is needed, use the schema in [program schema](references/program-schema.md).

## Gate new experiments

Before recommending or creating a substantial experiment, answer:

1. Which active manuscript claim or reviewer risk does it address?
2. What uncertainty remains after considering existing and historical evidence?
3. What outcomes would strengthen, weaken, or leave the claim unchanged?
4. What decision will each material outcome trigger?
5. Which paper figure, table, or paragraph would consume the result?
6. What is the cheapest valid test, and what is its stopping condition?

If these answers are not concrete, search existing analyses and history before proposing
new work. If the experiment would not change the paper, put it in a parking lot or reject
it. Curiosity alone is not sufficient reason to expand the active program.

Prefer closing an evidence gap, resolving a contradiction, or writing supported claims
over opening another direction. Keep at most a small number of unresolved campaigns
active; recommend pausing lower-value work when the program is diffuse.

## Define progress correctly

Use this lifecycle for paper-directed work:

`proposed -> admitted -> running -> analyzed -> claim decided -> integrated`

- `analyzed` means the result is valid and interpreted.
- `claim decided` means its consequence for the paper is explicit, including a negative
  or null result.
- `integrated` means the manuscript, claim ledger, figure/table plan, or explicit
  rejection record incorporates that consequence.

An experiment is not complete at `analyzed`. Do not report a growing experiment count
as paper progress. Report closed claim gaps, resolved reviewer risks, integrated panels,
and remaining blockers.

## Choose the next action

Rank candidate actions by expected paper value:

1. correctness threats that could invalidate a central claim;
2. missing evidence without which the thesis cannot stand;
3. contradictions or obvious reviewer objections;
4. integration of already-supported results into figures and prose;
5. robustness that materially changes credibility;
6. exploratory extensions.

Account for cost, time, dependency risk, and whether an existing result already answers
the question. Recommend one primary next action and explain why alternatives are lower
priority. When core claims meet their evidence threshold, default to writing and
submission preparation; additional experiments require a specific paper-level reason.

## Work across the two repositories

- Treat experiment configs, code, committed results, and detailed findings in PRODIGY
  as the evidence source.
- Treat manuscript choice, paper claims, narrative, contribution boundaries, and
  submission state in the paper repository as the paper source.
- Link claims to exact committed evidence rather than copying mutable conclusions
  between repositories without provenance.
- Use `prodigy-experiments` for experiment layout and analysis conventions.
- Use `prodigy-tucker` for cluster execution.
- Read `docs/agent_guides/repo_traps.md` before admitting evaluation evidence.

Do not modify both repositories merely to keep them cosmetically synchronized. Make the
smallest authoritative update and record exact evidence locations and revisions.

## Reporting

For planning or status requests, report:

- active manuscript or unresolved manuscript choice;
- thesis and central claims affected;
- what is actually supported versus only hoped for;
- the binding paper blocker;
- the single highest-value next action;
- work to stop, pause, or park;
- the concrete manuscript integration target and completion condition.

Favor an honest narrower paper over a broad story sustained by unresolved experiments.
