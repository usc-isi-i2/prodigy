# Complete source-pair composition analysis

This analysis is registered before the complete source-pair results are
available. It validates the 36 pairs by three seeds by nine targets Cartesian
product, then compares three prespecified constituent predictors:

- the target-specific best included specialist;
- the mean of the two included specialists;
- the included donor selected by leave-one-target-out global donor quality.

The main validation subset contains targets held out from pairs that are not the
rung-two set of any original nested ladder. The best-specialist envelope is
called replicated only when it has lower MAE than both alternatives in every
training seed and its target-demeaned correlation is positive in every seed.
This panel holds source count fixed at two and therefore cannot establish a
source-count scaling law.

The Tucker postprocessor writes raw derived tables, a summary JSON, and PNG/PDF
figures beneath the experiment run root. Results enter this analysis directory
only after the 972-cell evaluator audit passes.
