# Role interaction: correction to the manuscript's removal narrative

Re-audit of completed data, 7 September 2026. No new model results in this note.

The paper says blanket background-edge removal would miss a support-specific
repair. That is contradicted by the existing Hong Kong factorial: removing
both roles gives higher AUC than removing supports alone on both political
streams. The correct finding is a conditional role effect, not evidence that
support-only removal is the best of the tested interventions.

| Target / HK checkpoint | Original: intact / query removed / support removed / both removed | Second: same order |
|---|---|---|
| covid-political | .7489 / .6764 / .8552 / .8646 | .7932 / .7265 / .8853 / .8918 |
| election2020 | .9333 / .7122 / .9701 / .9820 | .9586 / .7139 / .9623 / .9731 |

Hong Kong query-only removal harms AUC, yet its incremental effect after
support removal is positive on both streams of **four targets**: covid-political,
election2020, facebook-page-reference and twibot20. Joint removal improves over
baseline on the first three but still harms twibot20. Thus conditional recovery
after another harmful change must not be advertised as net improvement.

This pattern also appears beyond Hong Kong. Requiring the same sign reversal
on both streams, the foreign-source counts are 2/8 on covid-political, 3/8 on
election2020, 3/8 on facebook-page-reference, 4/8 on twibot20, and 0/8 on
ukraine-suspended. These are exact sign counts, including small effects; they
are not significance claims or independent training replications. The script
exports all individual magnitudes and accuracy/NLL contrasts as well.

The supported descriptive claim is: **the measured benefit of query-side graph
processing depends on how supports are processed.** AUC nonadditivity alone
does not establish nonadditive internal computations, and these results do not
identify an upstream property of the source corpus. The new role-topology
experiment tests the same conditional effects across all six existing HK/UKR
seed controls, with degree-preserving null graphs and representation geometry.

## Evidence

`audit_role_interactions.py` independently groups all four removal conditions
from `data/role_context_cells.csv`. It rejects duplicate/missing conditions and
changed checkpoint, weight, episode or query-count identities. All 90 complete
model/target/stream factorials are exported to `data/role_interaction_reaudit.csv`;
the input digest and complete source lists are in the adjacent JSON.

This re-audit changes the next paper revision: replace the claim that blanket
removal misses the gain, lead with the full factorial rather than only the
opposite-signed isolated ablations, and treat label-vector reconstruction as
implementation verification. The PDF has not yet been changed by this note.

Branch `codex/role-topology-interactions`; local worktree
`/Users/philipp/projects/gfm/prodigy/.worktrees/role-topology`. Original evidence
remains unchanged in `codex/target-performance-mechanisms`.
