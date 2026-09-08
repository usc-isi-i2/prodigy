# Research synthesis: support context constructs the classifier

## Central argument

Transfer is usually evaluated as a property of the representation a pretrained
model gives to a new example. Graph in-context prediction introduces another
object that must transfer: the computation that turns labeled supports into a
classifier. The same graph encoder participates in both computations. A target
can therefore have useful query representations and still receive an unsuitable
classifier from its support neighborhoods.

This distinction changes what a transfer analysis should measure. Input distance
and source size describe the data available to pretraining. Linear probes test
whether target labels remain recoverable. Neither alone tests whether the
learned support-to-class computation uses that information successfully. We
study that computation directly by holding the target episodes and pretrained
weights fixed and intervening separately on support and query roles.

## Evidence that carries the explanation

The most informative experiment separates two effects of support context on the
metagraph: its keys determine attention weights, while its values determine the
content aggregated into class representations. In the studied one-layer
architecture, transplanting only support values can leave queries and attention
unchanged. This intervention tests whether changed class-reference content is
sufficient to alter discrimination without improved query encoding or a new
attention allocation.

On the Hong Kong-to-political discovery model at 2,500 updates, value replacement
improves within-episode AUC by 5.66 points, compared with 1.23 for key replacement.
The prediction that the value intervention will improve discrimination and
exceed the key intervention is then frozen. At the nominated 50k checkpoint,
value gains are 6.86 and 8.93 points on the two episode streams, compared with
0.15 and 0.84 for keys. Joint replacement reproduces the removed-context endpoint
(8.76 and 10.35 points). This is not an additive variance decomposition: the
joint intervention includes interactions between changed keys and values.

The result identifies a computational pathway and makes a successful prediction
beyond the discovery recipe. It does not establish that value norms, value
directions, oversmoothing, graph density, or source semantics are the root cause.
The 50k model changes more than training duration, and its source-target
relationship is the same. These are limits on generality, not reasons to discard
the successful component prediction.

## Why the practical story must change

The strongest current simple deployment rule in the singleton panel is the U1
ridge ensemble, which bypasses the learned metagraph. It exceeds TRACE fusion
in accuracy, macro-F1, and AUC. TRACE shows that retaining predictions consistent
with a support readout improves a full-model ensemble; it does not demonstrate
that filtering full-model predictions is preferable to using the readouts.

The completed source-allocation experiment also fails its primary comparison.
Agreement-guided allocation gives .6523 accuracy/.7111 AUC against uniform
interleaving's .6589/.7155 on the nominated Facebook target. This rules out
presenting the tested allocation rule as the practical benefit of the mechanism.

Moreover, improved ranking is not equivalent to improved classification. The
earlier 50k political context-removal analysis reports an accuracy collapse even
as AUC improves. Removing support context is therefore a diagnostic intervention,
not a universal inference repair. The paper should retain the prediction-rate,
accuracy, and F1 evidence beside discrimination results.

## What the community would understand differently

Graph context has two roles in in-context transfer: representing the item to be
classified and constructing the classifier from the supports. An intervention
that helps one role need not help the other. Consequently, a shared treatment of
support and query neighborhoods is an architectural assumption to test, rather
than an automatic consequence of using one encoder. The present evidence
motivates role-aware model design; it does not yet establish a superior design.

This is a sharper claim than either 'representations are good but inference is
bad' or 'more graphs can hurt.' Representation quality varies across targets,
and context can be beneficial. The proposed contribution is the separation and
predictive localization of these roles, together with an evaluation that can
distinguish class-reference effects from query effects and operating-point shifts.

## Next decision

The public Wiki-to-FB15K-237 native reproduction is underway. It supplies a
different domain, multiclass task, and two-layer metagraph. Its role intervention
must be implemented faithfully; the one-layer fixed-query property cannot be
assumed. A successful native reproduction followed by the frozen component
test would materially strengthen the intellectual case. It would still be one
public transfer setting, not evidence of universality.

No new selector sweep is justified before that result. If the public mechanism
fails, revise the explanation's domain of validity rather than replacing the
intervention with whichever variant improves the same target.

## Provenance

The K/V numbers above were checked against Tucker's
`/dataMeR1/phil/gfm/prodigy-classrefkv/log/classrefkv_long_20260907/`
`protocol.json` and `prediction_assessment.json` (runtime revision `5ffbfa2b`).
The underlying `class_reference_kv.py` enforces support-only transplants,
single-layer evaluation mode, and attention/value reconstruction checks.
The 50k collapse is documented in the existing class-reference-value-path
decision brief. Allocation metrics are in this directory's `data/` exports.
This synthesis supplies an argument for revision; it is not a completed paper
or a claim that the publication goal has been met.
