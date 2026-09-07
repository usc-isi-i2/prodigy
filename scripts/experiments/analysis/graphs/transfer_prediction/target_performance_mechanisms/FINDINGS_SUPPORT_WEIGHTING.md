# Class weighting does not explain the support-validation reversal

7 September 2026. Post-hoc rescoring of existing natural-support predictions;
zero new model forwards, unchanged selected contexts, no prospective claim.

The earlier study used balanced leave-one-pair-out support CV but naturally
sampled queries. Test whether equal query-class weighting removes its opposite
within-episode support-CV/query-loss relationships for Hong Kong and Ukraine.

Read all 12 saved political prediction tensors (two sources, three seeds,
two streams), eight contexts and 128 recipient episodes each. Recompute
float64 query cross-entropy and reconcile all original episode/draw losses
with `episode_scores.csv` within 1e-5. Confirm saved selected contexts equal
the original CV argmin. Both query classes occur in every episode. Replace
only the query scoring weights by one half per local class; local class
permutations therefore do not affect this balanced score. Do not choose new
supports or alter logits, weights, labels or the observed query population.

Center both CV and query losses across eight draws within each recipient
episode before computing the descriptive Pearson correlation. These values
do not provide independent-query significance or new held-out confirmation.

| Source | Stream | Seed | Natural correlation | Balanced correlation | Selected-minus-random balanced NLL |
|---|---|---:|---:|---:|---:|
| Hong Kong | original | 0 | .33089 | .22241 | -.08013 |
| Hong Kong | original | 1 | .59576 | .53366 | -.39660 |
| Hong Kong | original | 2 | .27724 | .15995 | -.00155 |
| Hong Kong | fresh | 0 | .41981 | .32423 | -.12450 |
| Hong Kong | fresh | 1 | .60157 | .56001 | -.33739 |
| Hong Kong | fresh | 2 | .22365 | .14128 | +.00575 |
| Ukraine | original | 0 | -.03058 | -.05523 | +.03850 |
| Ukraine | original | 1 | -.20687 | -.21707 | +.04552 |
| Ukraine | original | 2 | -.16896 | -.18609 | +.05070 |
| Ukraine | fresh | 0 | -.02655 | -.03658 | +.00225 |
| Ukraine | fresh | 1 | -.17233 | -.14718 | +.03425 |
| Ukraine | fresh | 2 | -.16487 | -.16687 | +.03917 |

The opposite association signs survive in all six comparisons per source.
Equal class weighting alone is therefore not sufficient to explain the
reversal. Hong Kong's selected-context benefit falls from six of six under
natural NLL to five of six under balanced NLL; Ukraine worsens in all six
balanced-NLL comparisons. Neither calibration nor distributional differences
within each class have been isolated by this check. This is not evidence for
a universally valid support selector or a predictor of edge-suppression utility.

Advisor decision: do not propose class weighting as the repair and do not
restart a selector study. Preserve the model-dependent nature of support
validation in the contribution assessment; continue to distinguish natural
support choice from graph-message interventions.

Source: `/dataMeR1/phil/gfm/prodigy-mechanisms-role/log/natural_support_full_20260906/`
`predictions/{original,fresh}/covid_political/*.pt`, matched to the same run's
`episode_scores.csv`. SHA256s were computed for the 12 consumed files during
the audit. This private note is uncommitted in local worktree
`/Users/philipp/projects/gfm/prodigy/.worktrees/role-topology`, branch
`codex/role-topology-interactions`; no manuscript or PDF was changed.
