# Early support benefit is not concentrated in one dominant value contribution

7 September 2026. Exploratory saved-state analysis; no new model forwards,
fitting, or label-dependent support selection. Private evidence.

Question: can one or two geometrically dominant supports reproduce the early
fixed-attention value effect? This does not test every possible sparse subset,
nor does it identify misannotated or semantically misleading examples.

Use all 32 original saved early K/V batches, 128 episodes, 20 supports and
24 queries per episode. Recover episode membership and local class order from
the original batch. Hash-check those inputs against the producing receipts.
For each support, transport its removed-minus-intact U1 displacement through
the verified fixed-attention map `D O A_cs V`, yielding its additive change to
the two unnormalized class references. Keep final queries fixed. Rank supports
by Frobenius norm of this pair of reference changes, without using query labels.
Read out intact plus the largest one, largest two, or all contributions.

All-support reference reconstruction differs from saved value-only references
by at most 2.692e-6 per coordinate; its aggregate AUC matches the existing
value-only result. This is numerical reconstruction, not bit-exact replay.

| Saved-state readout | Mean within-episode AUC |
|---|---:|
| Intact | .841073 |
| Largest contribution only | .852720 |
| Largest two contributions | .858290 |
| All contributions / values only | .897714 |

The largest contribution carries a mean 10.72% of the sum of individual
contribution norms. The norm-based effective number of contributors,
`(sum norm)^2 / sum(norm^2)`, averages 13.57 of 20. This is not attention
entropy or a causal fraction of AUC explained.

Changing a single support helps AUC for 5.25 supports per episode on average
and hurts for 4.64; the remaining changes leave within-episode ranking unchanged.
The best singleton *chosen after scoring* gains 2.99 points on average; that
oracle quantity is descriptive, not a feasible selector. All-support updates
help in 86/128 episodes. In only 13 of those 86 does the largest-norm singleton
match or exceed the all-support gain. The mean top-one/top-two gains are
1.16/1.72 points versus 5.66 for all values; do not express these as additive
mediation percentages because normalized readout and AUC are nonlinear.

## Interpretation

Reject the narrow claim that one geometrically dominant support generally
accounts for the early value-path benefit. Contributions are distributed,
and individually helpful and harmful changes coexist. This does not prove
that all supports must change, rule out a specially chosen sparse subset,
or explain why the collective change improves class discrimination.

Do not start subset search, oracle support cleaning, or another selector study.
The result directs explanation toward how the class reference combines
support variation, while preserving the prior finding that the early fixed
prototype also improves. A constructor-only failure remains unproven.

Sources: Tucker `prodigy-classrefkv/log/classrefkv_discovery_20260907/`
`private_activations/original/batch_*.pt`, exact original inputs resolved by
`load_cell(phase='discovery')`, and recorded early checkpoint. Executed CPU-only
from `prodigy-label-context-discovery` at 9e19e5eb. Local private report in
`.worktrees/role-topology`, branch codex/role-topology-interactions.
