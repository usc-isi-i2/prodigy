# Fixed temporal role-bridge decision

7 September 2026, before mixed-role outcomes. Private analysis decision;
implementation is being prepared separately. No new training or target selection.

## Question

Does the replicated adverse late-encoder effect enter primarily through support
inputs to class-reference construction, or through query representation inputs?
This is distinct from existing fixed-checkpoint context-removal route tests.

## Fixed comparison

Use public Wiki → FB15K-237 seeds 0 and 1, the same first 128 saved fresh episodes,
and exactly the late (8,001-update) inference state. Define A(S,Q), where S and Q
select early (2,001-update) or late (8,001-update) pre-metagraph data rows.
Evaluate EE, EL, LE, LL. Early/late choices refer only to the data rows' encoder
checkpoint, not label initialization, inference weights, or inference buffers.

EE must reproduce E2000I8000; LL must reproduce E8000I8000 from the completed
whole-module crossover. The two mixed conditions per seed are the only new
scientific cells. Re-running endpoints is a parity check, not additional evidence.
Verify natural late pre-M output against saved late output before intervention;
then use saved donor rows exactly. Support and query indices must exhaust data
rows without overlap and agree with native query-mask ordering. Preserve labels,
graph inputs, inference state, RNG, module modes, and current label rows.

## Contrasts and interpretation

Primary outcome is accuracy, with macro-F1, AUC and NLL reported for every cell.
No target label is used to choose donors or interventions. Report each seed
separately and the paired episode contrasts:

- Late support at early queries: A(LE) − A(EE).
- Late support at late queries: A(LL) − A(EL).
- Late queries at early supports: A(EL) − A(EE).
- Late queries at late supports: A(LL) − A(LE).
- Interaction: A(LL) − A(LE) − A(EL) + A(EE).

The bridge prediction is negative support contrasts at both query endpoints in
both seeds. Support dominance additionally requires a larger average support
cost than query cost in each seed; report magnitudes rather than equating any
negative support contrast with dominance. Mixed signs, a larger query cost, or
large interaction weaken or reject a simple support-dominant explanation.
Do not change the target, checkpoints, seeds, metric, or donor definition after
seeing results. Do not launch follow-up partition sweeps to rescue that account.

## Scope limits

Pre-metagraph input localization is not final-query localization. The two-layer
metagraph and train-mode normalization can couple support, query, and label rows;
unchanged input query rows can still yield changed final queries. A positive
bridge result does not by itself reproduce the social fixed-query/value mediation.
Mixed-age states may also be off the naturally trained joint distribution, so
report their interaction and do not call conditional differences additive causal
fractions or infer preservation of task information.

No new fitted classifier, adapter, alignment, or selection method is proposed.
The separately prepared GILT comparison addresses family scope, not this role
question, and must not be conflated with it.

## Implementation preflight

Independent read-only inspection of seed-one episode 0 confirms both saved
trajectory donors have shape [140, 256]. There are 60 support and 80 query
rows, interleaved rather than contiguous; the native 140-by-20 query mask is
constant across class columns. At both checkpoints, saved query truth equals
the original labels indexed by that mask. The recorded task indices agree with
the interleaved role layout. This is a schema preflight on one episode, not a
substitute for the runner's full per-episode mask and receipt validation.

The runner is being developed in `/private/tmp/prodigy-publickg-temporal-roles`,
branch `codex/publickg-temporal-roles`, based on 6bfda494. No model run, commit,
or push has been performed for it at this checkpoint.

## Completed outcome: support-dominant bridge rejected

Both fixed seeds completed at d10aab89. Accuracy percentages, with late inference
fixed throughout:

| Support encoder | Query encoder | Seed 0 | Seed 1 |
|---|---|---:|---:|
| Early | Early | 78.0957 | 75.3516 |
| Early | Late | 50.6055 | 44.1895 |
| Late | Early | 51.0254 | 39.9121 |
| Late | Late | 75.0391 | 73.4082 |

The prediction fails in both seeds. Late supports hurt early queries by
27.0703 / 35.4395 points, but HELP late queries by 24.4336 / 29.2188 points.
The interaction is 51.5039 / 64.6582 points, far larger than the matched
early-to-late endpoint change. Neither role has a checkpoint-independent cost.
This does not provide the proposed bridge to harmful support-reference inputs.

The direct finding is severe sensitivity to mixed-age support/query inputs.
Coordinate mismatch or changed joint feature statistics are plausible explanations,
not established mechanisms; the artificial mixed states may be off-distribution.
Do not reframe this failed localization as proof that coordinate drift explains
the naturally occurring matched-checkpoint deterioration. Keep the social
fixed-query/value intervention and public temporal crossover complementary.
Per the stopping rule, no new alignment or role-partition sweep is justified
to rescue the support-dominant prediction.

Independent checks: 512 rows and 128 unique ordinals per condition per seed;
all 16 means per seed recomputed to 1e-12. Maximum endpoint logit errors are
7.6294e-6 and 1.5259e-5, below the fixed 1e-4 tolerance. Both completion statuses
are complete. These checks do not replace a fresh tensor-level re-score.

Tucker root: `/dataMeR1/phil/gfm/prodigy-publickg-temporal-roles/log`.
Summary SHA256, `roles_seed0_20260907/summary.json`:
`3d16668fd66e10aceca585e8a66d6a9e7d9eadc7f27038d8021c2d0d5a1e1da6`.
Seed 1, `roles_seed1_20260907/summary.json`:
`281b40dbdd1c8fc5f16d67e5dc337895e1bacabf5834deba88c883c13e3a7806`.
Code/tests only pushed on `codex/publickg-temporal-roles`; these findings remain
private and uncommitted on `codex/role-topology-interactions`.
