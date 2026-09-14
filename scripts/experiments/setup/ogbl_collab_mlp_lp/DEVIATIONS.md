# Operational deviations

This log is append-only. It records events that affect how campaign evidence
must be interpreted without silently rewriting the frozen protocol.

## 2026-09-13: smoke test exposed held-out scores

The full-data, one-epoch CPU smoke at producing revision `5eec95f1` exercised
the complete output path and therefore generated official test metrics after
both smoke-arm selections were frozen. The output was inspected to establish
that the job completed. This happened after the production arms, seeds,
optimizer, budgets, selection rule, official metric, and diagnostic strata had
already been frozen in the same revision.

No scientific setting was changed in response to the smoke scores. The smoke
artifacts under `/dataMeR1/phil/gfm/ogbl_collab_mlp_lp/smoke_cpu_seed0` are
classified as smoke-only and are excluded from aggregation. Production uses
the same training and evaluation code. Nevertheless, the official test panel
is no longer literally unseen by the experiment operator, and final findings
must disclose that qualification.
