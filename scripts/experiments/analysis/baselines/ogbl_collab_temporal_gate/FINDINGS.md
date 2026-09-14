# Temporal residual gate pilot

## Direct-score follow-up: completed, similar mean with lower observed seed spread

The unanchored 14-input, 513-parameter scorer reaches **67.7990%** mean official
2018 Hits@50 (sample SD 0.0710 percentage points), versus warm_v1's 67.8572%
(SD 0.6541 points) and official-calibrated AA-DC's 67.3557%. All three direct
seeds beat that AA-DC reference, but the mean is 0.0583 points below warm_v1.
The narrower three-seed spread is descriptive, not statistical proof of stability.
No HyperFusion win is established.

| Seed | Selected step | 2018 Hits@50 | Recovered / lost vs official AA-DC | Hits among 12,185 zero-base positives |
| --- | ---: | ---: | ---: | ---: |
| 0 | 400 | 67.8617% | 1794 / 1490 | 12 |
| 1 | 400 | 67.8134% | 1913 / 1638 | 12 |
| 2 | 400 | 67.7219% | 1793 / 1573 | 11 |

The original motivation was that these 12,185 positives have a zero frozen-AA-DC
base; at the warm gate's selected strength and observed negative cutoffs they
cannot be rescued even by its maximum positive correction. Removing that hard
anchor allows recovery but rescues only 11--12 under this budget. Thus the anchor
was not the only practical obstacle. This does not prove the input features lack
signal: every direct seed selected step 400, the final available checkpoint, and
the BCE objective, capacity, optimization and input representation remain possible
limitations. No convergence claim or automatically expanded training followed.

The direct scorer recovers 1,647 official-AA-DC misses consistently across all
three seeds, while consistently losing 1,346 hits. Unlike warm_v1, repeat pairs
improve slightly (+106 / +111 / +109 net hits); new pairs gain +198 / +164 / +111.
There are 22 distinct negatives entering the top 50 across seeds (13 shared by
all three); overall ranking still reflects a recovery/loss trade-off.

The raw logit is learned directly from the original 13 features plus
`log1p(frozen_2015_calibrated_AADC)`. It has no base addition, residual tanh bound,
or eligibility mask. The appended feature is standardized on 2016 with the other
inputs. Full warm_v1 training, identical BCE/optimizer/mining settings and 400-step
budget are retained. Input width adds 32 weights. Checkpoint selection uses full
2017, every 20 steps, earliest ties; no strength search and no baseline fallback.
Stored strength 1 is only a compatibility sentinel. These changes constitute a
scoring-family comparison, not a strictly one-parameter ablation.

Eight focused tests passed locally and the Tucker prelaunch test command passed.
The independent audit verifies all checkpoint/selection hashes, direct-only
checkpoint argmax and tie rules, all full-panel metrics, zero-base recovery counts,
and the self-pair rule. The 2015--2018 panels and original features exactly match
warm_v1; official pairs/features also match recorded fingerprints. All expected
seeds completed before downstream interpretation. There was no baseline substitution.

Producing revision `110565d0b58f37b5311e2ac4c18f225582d2a743`, branch
`codex/collab-temporal-gate`; local worktree
`/Users/philipp/projects/gfm/prodigy-collab-gate`, Tucker worktree
`/dataMeR1/phil/gfm/prodigy-collab-gate`. Runtime on Tucker:
`/dataMeR1/phil/gfm/ogbl_collab_temporal_gate/direct_v1`. CPU eight threads,
76.54 seconds, offline W&B; dedicated tmux `collab-temporal-gate-direct` exited
normally. Other concurrent Collab jobs ran in a different worktree and were not
modified. No GPU use or test scores here; the existing dataset identity audit
continues reading full-split metadata as declared. 2018 remains repeatedly
inspected development data, not a fresh holdout.

Evidence: [results](data/direct_v1/results.json), [protocol](data/direct_v1/protocol.json),
[training pool](data/direct_v1/training_pool.json), [selections](data/direct_v1/selection_frozen.json),
[receipt](data/direct_v1/validation_receipt.json), [audit](data/direct_v1/audit.json),
[error slices](data/direct_v1/error_overlap.json).
Reproduce audit with `audit.py --runtime <direct_v1> --evidence <direct_v1>
--control-runtime <warm_v1> --out <new-audit.json>`.

## Constrained-rescue follow-up: completed, negative result

Restricting corrections to new pairs with zero AA, and training only within that
group, did not improve performance. Mean official 2018 Hits@50 is **66.5868%**
(sample SD 0.4382 points), versus warm_v1's 67.8572% and official-calibrated AA-DC's
67.3557%. The unchanged frozen 2015-calibrated base is 66.8398%.

| Seed | Selected step / strength | 2018 Hits@50 | Recovered / lost vs frozen base |
| --- | --- | ---: | ---: |
| 0 | 400 / 0.25 | 66.0808% | 1561 / 2017 |
| 1 | 0 / 0 | 66.8398% | 0 / 0 |
| 2 | 0 / 0 | 66.8398% | 0 / 0 |

All three completed the 400-step budget. Seeds 1 and 2 selected the zero-strength
fallback because learned candidates did not beat it on full-panel 2017 selection.
Seed 0's selected correction failed to transfer: its negative cutoff rose by
0.586386 in log-score units. **1,601 protected positive hits were lost despite
their scores remaining bit-for-bit unchanged.** The constraint protects scores,
not ranks against the negative pool. This confirms the anticipated failure mode;
it does not prove all constrained methods must fail.

Compared with the separately official-calibrated AA-DC reference, seed 0 recovers
1,754 but loses 2,520 positive hits. Repeat/new net changes are -427 / -339.
The fallback seeds differ from that reference solely because calibration differs:
they recover 339 and lose 649 each. These are not learned improvements or damage.

The training pool contains 16,475 eligible 2016 positives and 99,957 eligible
negatives. All calibration, selection, assessment, graph, and feature inputs stay
as in warm_v1; all historical and official pair/feature arrays match the control.
The architecture remains 481 parameters, 3 seeds, 400 steps, same optimizer and
full-panel selection grid. This intervention jointly changes correction masking
and training-pool eligibility, so their separate contributions are not identified.

Six tests passed locally and the Tucker test command succeeded before training.
The independent audit verifies checkpoint and score hashes, selection rules, all
reported hit counts, input parity, self-pair handling, and exact protected-score
equality. A first local audit failed exact comparison to locally recomputed logs:
macOS versus Tucker NumPy log results differed by at most 9.54e-7. The audit now
uses the hash-verified zero-strength seeds as the exact producer-side baseline;
those two full score arrays match each other exactly and reproduce the frozen
AA-DC hitmask exactly. Protected-score equality is still exact, not tolerance-based.
No training artifact was changed to resolve this audit portability issue.

Producing revision `ed25a6bdbe34d4ead2e1176fa7292dee8e0e21a6`, branch
`codex/collab-temporal-gate`; local worktree
`/Users/philipp/projects/gfm/prodigy-collab-gate`, Tucker worktree
`/dataMeR1/phil/gfm/prodigy-collab-gate`. Runtime on Tucker:
`/dataMeR1/phil/gfm/ogbl_collab_temporal_gate/constrained_v1`. CPU eight threads,
77.34 seconds, offline W&B, no GPU use. Dedicated tmux
`collab-temporal-gate-constrained` exited normally. No test scores computed;
the existing dataset identity audit still reads full-split metadata as declared.
No additional sweep or test evaluation followed the negative result.

Evidence: [results](data/constrained_v1/results.json),
[protocol](data/constrained_v1/protocol.json), [training pool](data/constrained_v1/training_pool.json),
[selections](data/constrained_v1/selection_frozen.json),
[receipt](data/constrained_v1/validation_receipt.json),
[independent audit](data/constrained_v1/audit.json),
[error slices](data/constrained_v1/error_overlap.json).
Reproduce audit with `audit.py --runtime <constrained_v1> --evidence <constrained_v1>
--control-runtime <warm_v1> --out <new-audit.json>`.

Conclusion: do not replace warm_v1 with this variant. The current feature/model
combination still promotes difficult negatives too strongly; protected-score
masking alone plus subgroup training did not solve that discrimination problem.
This is post-hoc development evidence, not a fresh holdout or HyperFusion comparison.

## Warm-gate error overlap diagnostic

Post-hoc analysis of the unchanged official 2018 validation panel, using all three
frozen warm_v1 seeds and official-calibrated AA-DC as the reference:

- 840 AA-DC misses are recovered by every seed; all are previously unobserved
  author pairs, and 833 have zero AA (no common neighbor).
- 1,435 misses are recovered by at least one seed. Another 18,179 remain misses
  for AA-DC and every gate seed; 17,744 of these have zero AA.
- 433 AA-DC hits are lost by every seed, including 94 repeat collaborations.
  These consistent losses mostly have nonzero AA (427 of 433).

| Pair history | Positives | AA-DC hits | Net additional hits, seeds 0 / 1 / 2 |
| --- | ---: | ---: | ---: |
| Repeat collaboration | 28,072 | 27,947 | -77 / -234 / -264 |
| Previously unobserved pair | 32,012 | 12,523 | +827 / +370 / +282 |

Thus the gain is not merely memorizing repeat collaborations: every seed gains
on new pairs and loses on repeats. "New" means no previous pair event, not a
new author. AA-DC already hits 99.55% of repeat positives on this panel.

Define negative promotion as entry into the top 50 from outside AA-DC's top 50.
Seeds introduce 7 / 10 / 9 negatives respectively (14 distinct; three shared by
all seeds). Thirteen of the 14 have zero AA; none are repeat pairs. Negative
indices 29950, 46048, and 58383 enter all three top-50 sets, moving from AA-DC
ranks 105, 70, and 144 respectively into gate ranks 31--46. These are benchmark
negative labels, not externally verified absence of collaboration.

Consistent recoveries have median raw feature cosine 0.968 and minimum endpoint
degree 24; promoted negatives have median cosine 0.957 and minimum degree 14.5.
Consistent losses have median cosine 0.906 and minimum degree 4. These profiles
are descriptive correlations, not proof of which features caused score changes.
Each seed has a unique score at its 50th negative; membership uses fixed-index
tie breaking, while positive Hits retains the official strict comparison.
Top-50 turnover alone does not causally assign each lost positive to a negative.

The diagnosis supports investigating protection of already-strong structural
predictions and discrimination within the zero-common-neighbor population.
It does not establish that a particular protection rule will improve overall
ranking: changing scores also changes the negative cutoff. No rule was fitted,
ensemble selected, or additional model evaluated in this analysis.

Validation checks: assessment archive SHA-256 matches the committed receipt;
positive/negative pair and feature fingerprints match the run metadata; all
three full-panel Hits values and the AA-DC reference recompute exactly. No test
data were read by this diagnostic. Local worktree and branch remain
`/Users/philipp/projects/gfm/prodigy-collab-gate`, `codex/collab-temporal-gate`.

Evidence: [overlap and negative-rank report](data/warm_v1/error_overlap.json).
Reproduce with `error_overlap.py --runtime <warm_v1> --evidence data/warm_v1
--out <new-report.json>`. This is development evidence from repeatedly inspected
2018 validation, not an independent holdout or a causal feature ablation.

## Warm-endpoint follow-up: completed

The matched-population follow-up improves mean official 2018 Hits@50 from
67.5821% to **67.8572%** (+0.2752 percentage points), versus 67.3557% for
official-calibrated AA-DC (+0.5015 points). Sample standard deviation across the
three optimization seeds is 0.6541 points. This is a modest development gain,
not evidence of beating HyperFusion, and seed sensitivity remains substantial.

| Seed | Step / strength | 2018 Hits@50 | Recovered / lost vs official AA-DC |
| --- | --- | ---: | ---: |
| 0 | 400 / 0.25 | 68.6040% | 1294 / 544 |
| 1 | 360 / 0.25 | 67.5821% | 1025 / 889 |
| 2 | 340 / 0.25 | 67.3857% | 1006 / 988 |

All three exceed the official AA-DC reference, but seed 2 gains only 18 net
positive hits. Relative to the original gate, paired-seed changes are +0.6741,
+0.0666, and +0.0850 points. Do not select seed 0 retrospectively using 2018.
The common frozen AA-DC baseline remains 66.8398%; its recomputed calibration
and final score were unchanged by filtering.

The independent audit verifies exact equality of all historical negative pairs
and features, historical positive features as precisely the warm-endpoint subsets,
and unchanged official 2018 positive/negative inputs. Checkpoint and selection
hashes, selection argmax/tie rules, every final hit count, and self-pair scoring
pass. All three required cells completed with no expansion or test scoring.
The full dataset identity audit still reads test metadata, as declared in setup.

Producing revision `7d528e274fd969a0abba5af7d7619e9c384b2190`, branch
`codex/collab-temporal-gate`; local worktree
`/Users/philipp/projects/gfm/prodigy-collab-gate`, Tucker worktree
`/dataMeR1/phil/gfm/prodigy-collab-gate`. Runtime evidence is on Tucker at
`/dataMeR1/phil/gfm/ogbl_collab_temporal_gate/warm_v1`. CPU only, eight threads,
78.02 seconds; dedicated tmux `collab-temporal-gate-warm` exited on completion.
W&B offline directory is recorded in the receipt. No GPU jobs were started.

Evidence: [results](data/warm_v1/results.json), [frozen protocol](data/warm_v1/protocol.json),
[selections](data/warm_v1/selection_frozen.json), [receipt](data/warm_v1/validation_receipt.json),
[independent matched-input and score audit](data/warm_v1/audit.json).
Audit reproduction: `audit.py --runtime <warm_v1> --evidence <warm_v1>
--control-runtime <pilot_v1> --out <new-audit.json>`.

This intervention jointly changes historical calibration/training/selection
positive eligibility; it does not identify which stage contributes the gain.
Official 2018 is repeatedly inspected development data, not a fresh holdout.
The original pilot below is preserved unchanged as the control.

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
The positive populations also differ substantially. The follow-up below explains
the main positive-population difference. The 2017-to-2018 absolute score jump should not
be interpreted as training progress or directly comparable difficulty.

Before scaling this approach toward the HyperFusion target, the value of these
historical panels as a model-selection proxy needs to be established. This pilot
supports a small transferable correction, not a confident test-score forecast.
No additional sweep or test evaluation was launched after these outcomes.

### Follow-up: endpoint history explains the structural coverage gap

Read-only inspection of the original Tucker training and validation split files
confirms that 2017 contains 23,008 positives with neither endpoint previously
active, 49,068 with only one previously active endpoint, and 47,546 with both.
All 60,084 official 2018 positives have both endpoints previously active.
Activity here means at least one training edge strictly before the target year.
The raw CSV and training split have identical canonical pair-and-year multisets
(1,179,052 records), ruling out raw/split year misalignment in this check.

| Positive population | Count | Nonzero AA | Previously seen pair |
| --- | ---: | ---: | ---: |
| All historical 2017 | 119,622 | 25.92% | 19.28% |
| Historical 2017, both endpoints previously active | 47,546 | 65.21% | 48.52% |
| Official 2018 | 60,084 | 64.88% | 46.72% |

Thus 60.25% of historical positives have a cold endpoint, compared with zero
official validation positives. Conditioning on two previously active endpoints
reduces the AA-coverage gap from 38.96 points to 0.33 points (opposite direction).
This is a population mismatch, not evidence that learning improved dramatically
between years. Other differences, including negative sampling, remain.

[OGB documentation](https://ogb.stanford.edu/docs/linkprop/) specifies year cutoffs
but does not explain this endpoint-population difference. These observations are
consistent with warm-endpoint selection, but do not prove the upstream filtering
algorithm or its intent. No authoritative construction script was established.

The original pilot remains valid for its declared panels, but the unfiltered
historical panel is a poor population match to official validation. A warm-endpoint
historical diagnostic is justified; it would be a new, post-hoc protocol, not a
retroactive replacement of the recorded results. No retraining or test access was
performed for this audit.

Cached-feature counts and SHA-256 provenance are in [population audit](data/population_audit.json).
Reproduce with `audit_population.py --cache-root <pilot_v1> --output <output.json>`.
The raw/split metadata cross-check was a separate read-only Tucker inspection;
the cache-only helper does not independently repeat that check.

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
