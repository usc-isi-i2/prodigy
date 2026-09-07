# The late suppressed supports collapse before the metagraph

7 September 2026. Saved-state localization; zero new model forwards. This
changes the emphasis of the reference-compatibility hypothesis.

## Measurement

For every episode, recover support labels from positive support-to-label
edges. At U1, normalize each support, compute its squared Euclidean deviation
from the mean unit support vector, and average across supports. This dispersion
uses no query labels. For class separation, average unit supports within each
class, normalize the two means, and measure their Euclidean difference.
Also compute raw class-mean distance divided by mean support norm. Finally,
measure the distance between normalized final class references. Average each
quantity equally across episodes, not across mixed batches.

| Configuration / condition | U1 unit dispersion | U1 unit class separation | U1 raw relative class separation | Final reference separation |
|---|---:|---:|---:|---:|
| 2.5k original / intact | .263453 | .483574 | .433900 | .751406 |
| 2.5k original / suppressed | .153734 | .367055 | .351904 | .454122 |
| 50k original / intact | .145405 | .380692 | .408167 | .282174 |
| 50k original / suppressed | .000272 | .016210 | .019559 | .027320 |
| 50k fresh / intact | .145407 | .378023 | .405267 | .284739 |
| 50k fresh / suppressed | .000285 | .015501 | .018329 | .026601 |

Each pair covers all 128 original episodes or all 128 fresh episodes. This
uses the same hash-recorded activation sets as
`FINDINGS_SUPPORT_PROTOTYPE_CONTRAST.md`. No query outcomes enter these
geometry measurements, but the configuration outcomes were already known.

## Interpretation that survives the measurement

Late suppression reduces pre-metagraph support unit dispersion by roughly
510-535 times. The supports are nearly indistinguishable *before* class
references are built. Early suppression reduces dispersion modestly and
retains substantial class separation. The late collapse cannot be attributed
solely to a metagraph turning well-separated supports into coincident references.

This does not overturn the prototype result: late suppression worsens U1
prototype ranking but improves current-interface native ranking. Rather, it
clarifies what is being traded away: nearly all angular support diversity is
lost, yet the remaining differences can improve a fragile native ranking.
Restoring training-label initialization strongly moderates that gain. The
interaction therefore involves an upstream collapsed input and downstream
initialization, not a proven isolated defect of the value projection.

The implemented S block contains graph convolution, normalization/ReLU and
a learned center-plus-neighborhood pooling/reset operation. These are all
upstream of the saved U1 point. This measurement does not isolate which
learned component creates the collapse, or show that training duration alone
causes it: early and late configurations differ in protocol. No statement
that an individual ReLU or BN operation is responsible is warranted yet.

## Consequence for the paper synthesis

Replace "learned reference construction collapses distinct support embeddings"
with "suppression can deliver nearly collapsed supports to reference
construction, making its observed benefit initialization-dependent." The
first phrase was a hypothesis, not a verified result. Retain the early
robust ranking-and-accuracy benefit and public counterexample separately.

The next substantive explanation, if pursued, concerns why this encoder loses
support diversity without graph messages, not another partition of final
class vectors. This finding supplies a location and a contrast; it is not yet
a causal account of the learned component responsible.

Private worktree `.worktrees/role-topology`, branch
`codex/role-topology-interactions`, code `9e19e5eb`. Inputs remain on Tucker
under `prodigy-classrefkv/log/{classrefkv_discovery_20260907,classrefkv_long_20260907}`.
