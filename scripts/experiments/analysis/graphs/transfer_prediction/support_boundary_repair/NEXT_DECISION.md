# Return to the pretraining question

The value-repair direction is closed for the nominated protocol. Do not replace
it with another tuned threshold on the same target. The publication still needs
a consequential mechanism or method, not a collection of localized failures.

## Competing explanations to distinguish with retained evidence

1. Source/schedule effects mainly change representation quality: source rankings
   persist under the same U1 ridge readout. Remedy would concern pretraining data
   or encoder learning, not a repaired metagraph.
2. Source/schedule effects mainly change inference compatibility: U1 rankings
   differ from full-model rankings and full inference destroys usable decisions.
   Remedy would decouple reusable representations from a source-specific task
   solver, but must outperform U1, not just the original full model.
3. Apparent health predictiveness mostly reflects U1 being a better classifier.
   Agreement–accuracy correlation alone cannot discriminate this explanation.

Next low-cost analysis: reuse the27-model matched-example schedule exports, if
they retain U1 ridge predictions, to compare U1 performance, full performance,
and full-minus-U1 gains on identical queries across schedules. Include direct U1
ensembles in the old fusion comparison. No new training until this audit states
whether the existing evidence distinguishes these explanations.

Descriptive decision decomposition: full accuracy minus U1 accuracy equals
corrections of U1 errors minus corruptions of U1-correct examples, divided by query
count. This identity is useful bookkeeping, not causal proof. Report the two
counts rather than interpreting agreement as preservation by definition.

## Novelty boundary

Meta-learning through a differentiable ridge solver already exists:
https://arxiv.org/abs/1805.08136 (Bertinetto et al., ICLR2019).
Therefore replacing the metagraph with differentiable ridge is a meaningful
baseline/experimental contrast, not by itself a novel framework. Any proposed
training method must be justified by an identified graph-transfer failure and
demonstrate a benefit beyond existing solver-based meta-learning.

Historical schedule scope: the matched27-model2/3/4-source2500-update study did
not reproduce a universal pair-to-many phase change. The older eight-source40k
result remains protocol-specific. Do not present the simple source-count law as
established, and do not reuse superseded static-link-prediction synergy claims
from the old program synthesis.
