# Paper planning

Prose planning for the paper: theses, route options, directions, and related work.
These are *not* experiments — no configs or runners live here. Experiment setup stays
in `scripts/experiments/setup/`, findings in `scripts/experiments/analysis/`.

Consolidated 2026-07-26 from seven scattered folders under `scripts/experiments/setup/`
(`gap_to_paper-jul_21`, `gap_to_paper_v2-jul_22`, `coverage_not_composition-jul_21`,
`paper_routes-jul_21`, `paper_stories-jul_21`, `paper_directions-jul_25`,
`related_work_gaps-jul_25`).

## What to read first

| Doc | What it is | Status |
|---|---|---|
| [`v2_channels_jul22.md`](v2_channels_jul22.md) | *Which Graphs for Which Objectives?* — channel-aware joint mixtures. Full thesis + execution plan. | **Current thesis** |
| [`directions_jul26.md`](directions_jul26.md) | Open research directions organised by which knob they turn (data / objective / sampler / representation / scale / eval / generality). | **Current question list** |
| [`directions_jul25.md`](directions_jul25.md) | Structured direction options, scored against the literature. | **Current shortlist** |
| [`related_work/RELATED_WORK_AND_GAPS.md`](related_work/RELATED_WORK_AND_GAPS.md) | Related work synthesis + open lanes; per-lane detail in [`agent_reports/`](related_work/agent_reports/). | Current |
| [`lit_gap_analysis_jul25.md`](lit_gap_analysis_jul25.md) | Verified gap analysis: which lanes are open, which are scooped, deadlines. | Current |
| [`state_doc_jul22.md`](state_doc_jul22.md) | Draft abstract/intro/method skeleton with dataset tables. | Working draft |

## Earlier drafts (kept for provenance)

| Doc | What it is |
|---|---|
| [`v1_coverage_jul21.md`](v1_coverage_jul21.md) | *Coverage Without Composition?* — the coverage-focused v1 thesis. Still the best written account of the neighbor-matching evidence; superseded as a paper plan by v2. |
| [`scratch_jul21.md`](scratch_jul21.md) | The original "naive paper" scratch note both v1 and v2 build on. |
| [`directions_jul21.md`](directions_jul21.md) | First unstructured brainstorm of paper stories. Superseded by `directions_jul25.md`. |

## Routes

Three worked-out framings of the same evidence, plus the scratch notes they grew from.

| Doc | Framing |
|---|---|
| [`routes/combined.md`](routes/combined.md) | Channel alignment as the causal spine joining the ML and CSS stories. |
| [`routes/css.md`](routes/css.md) | Coverage, transfer, and blind spots across online events. |
| [`routes/ml.md`](routes/ml.md) | Capability as a property of the objective set. |
| [`routes/scratch_jul21_ml.md`](routes/scratch_jul21_ml.md), [`routes/scratch_jul21_ccs.md`](routes/scratch_jul21_ccs.md) | Earlier one-page versions; different headline claims, kept because they are not strict subsets. |

> **Caveat on `routes/ml.md`.** Its central claim — that the three-way objective
> rotation produces a structural capability absent from every pair — rests on the
> static-LP result that the 2026-07-23 evaluator rescore overturned. Link prediction
> is a neighbor-matching main effect, not a synergy. See
> [`multitask_ssl/FINDINGS_rescore.md`](../../scripts/experiments/analysis/multitask_ssl/FINDINGS_rescore.md).
> The route needs a new spine before it can be used.
