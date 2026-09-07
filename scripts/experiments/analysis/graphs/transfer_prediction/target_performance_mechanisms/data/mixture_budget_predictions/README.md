# Complete saved-update budget diagnostic

Unedited compact JSON outputs from Tucker
`/dataMeR1/phil/gfm/prodigy-mechanisms-budget/log/target_mechanisms/mixture_budget_predictions_20260906`,
frozen code revision `1fee2c57`. The CPU-only job launched at 12:12:14 UTC on
September 6 and was verified complete by 12:14:56 UTC. It reads existing
prediction tensors; no model was retrained or forwarded.

All 810 distinct prediction cells, 1800 comparisons, 5400 error strata, 10
cached-input sets and 54 actual training configurations passed validation. All
step-2500 outputs reproduce the preceding complete complementarity analysis.
`training_budget_audit.json` ties each training recipe to its saved config hash,
completed-run log/result and exact checkpoint path. The explicit save schedule
is 100/300/900/2500 completed updates; batches contain four training episodes.

The fixed largest-saved-step rule within 2500 total updates uses 900 per pair
member and 300 per LOO member. The primary foreign Facebook LOO prediction
**failed on both streams**. The favorable TwiBot result is secondary and does
not replace it. `DONE.json` certifies complete computation and verification,
not success of the scientific prediction. See the parent directory's
`mixture_budget_validation.json` for the explicit failed primary endpoint.

Models use the historical sampler and one training seed. Ensembles retain
2x/8x inference capacity and forwards. Update/episode budgets do not equate
FLOPs or wall time; training inputs are not matched, and no causal interference
or general remedy is established. Both episode streams and all saved steps are
retained. Large tensors and source data remain on Tucker.
