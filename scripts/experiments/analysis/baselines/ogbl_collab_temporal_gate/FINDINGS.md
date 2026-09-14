# Temporal residual gate pilot

## Outcome relative to the objective

The 481-parameter correction improves forward-year validation ranking, but this
pilot does **not** establish a route to exceeding HyperFusion's reported 71.29%
test Hits@50. No test score was computed. The three-seed mean on official 2018
validation is **67.5821 +/- 0.3198%** (sample standard deviation across training
seeds), compared with **67.3557%** for AA-DC calibrated on that same validation
panel. The difference is **+0.2263 percentage points**, with two of three seeds
above that reference. This is not a material breakthrough toward the target.

The temporally matched comparison is stronger: the gate improves on its own
AA-DC base, calibrated only on 2015, from **66.8398% to 67.5821%** on 2018
(**+0.7423 points**, positive in every seed). That demonstrates some transfer of
the correction across years. The stronger AA-DC reference has the advantage of
calibration on 2018 and is therefore labeled separately.

## Frozen timeline and model

- 2015 labels calibrate AA-DC gate and L3 scale using a graph through 2014.
- 2016 labels train the correction using a graph through 2015.
- 2017 selects checkpoint and correction strength using a graph through 2016.
- All three selections are saved before extracting/scoring the 2018 assessment.
- 2018 uses the official validation pairs and graph through 2017.
- The official 2019 test panel is never scored.

The model is a one-hidden-layer residual MLP (13 inputs, 32 hidden units, 481
parameters), trained with BCE and a mixture of uniformly sampled and hard-mined
2016 negatives. Inputs include historical AA/L3, local structure, raw author-feature
cosine, previous pair event counts and recency, and recent endpoint activity.
The correction can increase or decrease the base log score. Every self-pair
receives a fixed bottom score independently of labels; no evaluation pair is
removed. All graph and edge-weight lookup boundaries were checked.

The supplied static OGB node features were retained. These are not historical
feature snapshots; the historical guarantee applies to graph events and learned
parameters. No MLP trained through 2017 was reused to construct earlier training
features. Upstream L3 preserves supplied endpoint order; no claim of symmetry is
made for that upstream component.

## Results

| Seed | Selected step | Strength | 2017 selection Hits@50 (%) | 2018 Hits@50 (%) | Gain vs frozen AA-DC (points) | Gain vs 2018-calibrated AA-DC (points) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 360 | 0.25 | 28.1411 | 67.9299 | +1.0901 | +0.5742 |
| 1 | 380 | 0.25 | 27.8870 | 67.5155 | +0.6757 | +0.1598 |
| 2 | 400 | 0.25 | 27.9338 | 67.3008 | +0.4610 | -0.0549 |

The frozen AA-DC baseline scores 27.5025% on 2017. All selected checkpoints choose
the smallest nonzero correction strength, 0.25. No seed was selected using 2018.
All seeds ran the fixed 400-step budget; this was a bounded pilot, not a convergence
claim. The last step was selected for seed 2.

Against the frozen AA-DC base, the models recover 785/549/320 positive hits while
losing 130/143/43, respectively. Against the 2018-calibrated reference, they recover
776/753/530 and lose 431/657/563. The small aggregate improvement therefore still
reflects a trade-off between rescued positives and lost hits at the negative tail.

The previous post-hoc self-pair rescue reached 67.8250% validation after selecting
its common weight on 2018. The present gate is lower at 67.5821%, but its selection
was frozen on 2017. These selection protocols differ; the new pilot is not a new
best validation score and should not be advertised as one.

## Historical panel mismatch

An audit of saved features found a substantial shift:

| Panel | Positives with nonzero AA | Positives previously seen as pairs | Negatives with nonzero AA |
| --- | ---: | ---: | ---: |
| Historical 2017 | 25.92% | 19.28% | 40 / 100,000 |
| Official 2018 validation | 64.88% | 46.72% | 59 / 100,000 |

Historical negatives are deterministic uniform unordered pairs excluding the
target year's positives; the official validation negatives are used as supplied.
The positive populations also differ substantially. The cause of the full panel
shift has not been established. The 2017-to-2018 absolute score jump should not
be interpreted as training progress or directly comparable difficulty.

Before scaling this approach toward the HyperFusion target, the value of these
historical panels as a model-selection proxy needs to be established. This pilot
supports a small transferable correction, not a confident test-score forecast.
No additional sweep or test evaluation was launched after these outcomes.

## Evidence and execution

- Producing revision: `94674c47acbd4356d9437625dc7d686a685fee3a`.
- Branch: `codex/collab-temporal-gate`.
- Local worktree: `/Users/philipp/projects/gfm/prodigy-collab-gate`.
- Tucker worktree: `/dataMeR1/phil/gfm/prodigy-collab-gate`.
- Runtime: `/dataMeR1/phil/gfm/ogbl_collab_temporal_gate/pilot_v1/` on Tucker.
- CPU, eight Torch threads, 86.52 seconds total; historical feature construction
  took 17.84, 19.32, 21.23 seconds for 2015--2017 and 20.42 seconds for 2018.
  Each small-model training run took about 1.34--1.36 seconds.
- W&B recorded offline; exact directory is in the receipt. No GPU jobs were started.
- All three expected seeds completed. Official AA-DC replay matches the recorded
  validation score to its printed precision. Model Hits@50 matches the OGB evaluator.
- Local audit verified all checkpoint hashes, the frozen selection manifest hash,
  the saved assessment score hash, earliest-step/smallest-strength selection,
  all reported assessment hit counts and the self-pair rule.

This is a post-hoc benchmark pilot. Although model selection preceded 2018 scoring
within this run, that official validation panel was already seen in prior work.
The three seeds are optimizer variations on shared data, not independent splits.

Sources: [frozen setup contract](../../../setup/ogbl_collab_temporal_gate/README.md),
[results](data/results.json), [receipt](data/validation_receipt.json),
[frozen selections and training curves](data/selection_frozen.json),
[independent score audit](data/audit.json), [audit implementation](audit.py).
