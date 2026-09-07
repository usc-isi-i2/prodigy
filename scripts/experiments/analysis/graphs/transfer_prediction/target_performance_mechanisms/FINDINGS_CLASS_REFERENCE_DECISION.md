# Decision: graph context constructs a learned classifier

**Subsequent result:** `FINDINGS_CLASS_REFERENCE_KV.md` records the completed
support key/value test. Its discovery-selected value-path ordering passes in
both 50k streams with attention and queries fixed. It also corrects the
nonlinearity interpretation of the radial interaction. The dated results below
are preserved; they are not the current final explanation or next-test status.

7 September 2026. This supersedes the **interpretation and next-test status** of
`FINDINGS_MESSAGE_SCALE_DECISION.md`, without discarding its completed scale
experiments, first snapshot, or prior decision pages. No further experiment is
queued. The main manuscript has not been expanded.

## Paper decision

Pursue the mechanism paper around **learned, role-dependent classifier
construction**, not a suppression method and not a general "direction, not
norm" explanation.

The immediate tests establish real within-episode discrimination changes.
In the initial models, final contrast orientation carries most of the pooled
AUC change. The prospective 50k test preserves political repair and bot harm,
but fails the page-repair prediction and breaks the simple pre-metagraph
direction-only account. Its native performance is still weak relative to
strong readouts. This is evidence of effects beyond 2,500 updates, **not** a
successful replication of every mechanism claim or closure of the
weak-implementation objection.

The contribution is the link between three observations:

1. Context has distinct query-representation and support-to-classifier roles.
2. Training can reverse support-context utility on the same target inputs.
3. Final class-reference geometry distinguishes discrimination changes from
   episode-dependent score scaling. Upstream geometry can interact rather
   than transfer as a direction-only rule.

The last step is a computational diagnostic, not a claim that cosine decoding
depending on direction is itself novel. The substantive evidence is the
fixed-query intervention, its learned sign reversal, and the boundaries
exposed by an out-of-configuration prediction test. A precise upstream
affine/nonlinear explanation remains unresolved; no universal sign predictor
or deployable intervention-selection method is established.

## 1. Ranking survives; it is not solely cross-episode scaling

The saved-prediction analysis covers 1,500 cells from 360 prediction files,
including all 900 completed scale cells and 600 saved training-step endpoints.
It performs **zero new model forwards**. Native pooled AUC reproduces exactly.
Within-episode AUC is an equal-weight mean over defined binary episodes;
undefined episodes are counted, not replaced with chance. Logit margins are
used to avoid softmax-rounding ties. Global binary labels are respected except
for the episodic page-class task, which uses its local class mapping.

Hong Kong step-2,500 removal effects (AUC points):

| Target | Pooled mean | Within-episode mean | Within signs across 3 seeds x 2 streams |
|---|---:|---:|---|
| covid-political | +7.715 | +6.066 | 6/6 positive |
| facebook-page-reference | +1.224 | +0.887 | 6/6 positive |
| twibot20 | -8.851 | -8.701 | 6/6 negative |
| election2020 | +0.379 | +0.651 | mixed |
| ukraine-suspended | +2.383 | +5.078 | positive or zero; native AUC near chance |

Pooled and within-episode AUC use different pairs. Their numerical difference
is **not** an estimate of a causal scaling contribution. The final-contrast
factorial below tests scaling directly.

## 2. Final contrast: discrimination versus episode strength

For each binary episode, normalize the final query and label vectors and form
`d = lhat_1 - lhat_0 = r u`. The margin is `tau * r * dot(qhat, u)`.
Hold final queries fixed and combine intact (A) and removed-message (B) parts:

| Score condition | Orientation | Strength |
|---|---|---|
| Intact | u_A | r_A |
| Removed | u_B | r_B |
| Orientation only | u_B | r_A |
| Strength only | u_A | r_B |
| Unit-strength intact / removed | u_A / u_B | 1 |

These are final-score counterfactuals, not necessarily realizable latent label
pairs or target-fitted classifiers. Positive episode-constant scaling cannot
change within-episode ranking or threshold decisions; those invariants pass
exactly. Zero/undefined contrast direction is rejected rather than invented.

Hong Kong final-contrast results, mean +/- **SD of 3 training-seed means**
(each seed first averaged over its 2 streams), in AUC points:

| Target / step | Joint removal, pooled | Orientation only, pooled | Strength only, pooled | Joint, within episode |
|---|---:|---:|---:|---:|
| Political / 2,500 | +7.715 +/- 2.374 | +7.093 +/- 1.615 | +1.168 +/- 0.536 | +6.066 +/- 1.004 |
| Pages / 2,500 | +1.224 +/- 0.206 | +1.068 +/- 0.140 | +0.308 +/- 0.102 | +0.887 +/- 0.295 |
| Bots / 2,500 | -8.851 +/- 1.841 | -8.726 +/- 1.798 | -0.406 +/- 0.236 | -8.701 +/- 2.638 |
| Pages / 100 | -0.750 +/- 0.439 | -0.768 +/- 0.353 | +0.216 +/- 0.277 | -3.410 +/- 1.720 |

Orientation carries most of the initial models' changes, including bot harm.
Strength has a smaller but nonzero pooled effect. Do not add these contrasts
as percentages of explained variance: the factorial is non-additive.

The analysis also includes Ukraine on all three final targets. Its pooled
orientation/strength/joint effects are -0.130/+0.017/-0.120 on political,
+0.238/-0.041/+0.196 on pages, and -5.260/-0.214/-5.353 on bots. Bot harm is
a counterexample to universal suppression, **not** to directional mediation.

## 3. Learned sign reversal on fixed target inputs

Hong Kong page support-removal effects from the saved native logits:

| Training step | Pooled mean | Within-episode mean | Pooled signs | Within signs |
|---|---:|---:|---|---|
| 100 | -0.750 | -3.410 | 6/6 negative | 5/6 negative |
| 300 | -0.240 | -0.846 | mixed | mixed |
| 900 | +0.338 | +0.260 | 5/6 positive | mixed |
| 2,500 | +1.224 | +0.887 | 6/6 positive | 6/6 positive |

Do not extend the pooled 6/6 step-100 claim to within-episode ranking. Figure C
uses native-logit margins; Figure B uses the reconstructed final contrast.
At early step 100 the removed class vectors nearly coincide, so float32-native
versus float64-reconstructed margins create some ranking ties. The largest
endpoint discrepancy is 0.0732 AUC points within episode and 0.00172 points
pooled; the mean reversal and all sign conclusions are unchanged. At step
2,500 within-episode endpoints agree exactly, and maximum pooled difference
is 0.000085 points. This explains the slightly different step-100 seed SDs
between the two analyses (1.686 native versus 1.720 reconstructed).

Fixed target topology, text, collection window, and label rules cannot alone
explain a within-checkpoint-series reversal. This does not imply those target
properties are irrelevant to their interaction with learned processing.

## 4. One prospective 50k test: partial survival and decisive failures

The user approved one checkpoint, nominated **before** inspecting outcomes:
the Hong Kong model whose source-validation-selected best state equals
`checkpoint/state_dict_50000.ckpt`. No alternative checkpoint was evaluated.

Preserved: recorded S/U/M architecture, 768-dimensional inputs, 256-dimensional
embeddings, SAGE, saved BatchNorm, one-hop fanout 100 / limit 2,000, 3 supports
per class, batch size 1. Binary diagnostic episodes use the existing target
test splits and native target-specific query counts, 128 episodes per stream.
The training-to-classification label interface follows the original repository
evaluation (`ignore_label_embeddings=True` for NM training, False for target
classification). It is not the recent 2-hop/10-shot recipe or the original
all-class Facebook benchmark. Recorded model construction was checked against
the training revision; the currently audited loader is not claimed bit-exact
with the entire historical implementation.

Predictions were written into the run protocol before the full run. All three
targets and both streams were retained. A 2-episode smoke checked execution
and invariants only; its performance was not inspected to select the test.

Mean AUC changes in points, **one checkpoint and two episode streams**:

| Target | Joint pooled | Joint within | Final orientation only, pooled | Final strength only, pooled | Pre-meta direction only, pooled / within |
|---|---:|---:|---:|---:|---:|
| Political | +6.341 | +9.556 | +1.373 | -0.557 | -8.718 / -6.268 |
| Pages | -1.026 | -2.368 | -1.587 | +0.763 | -0.968 / -1.099 |
| Bots | -15.602 | -19.135 | -16.726 | -0.804 | -11.150 / -10.528 |

Decision against the prospective predictions:

- **Political repair survives:** both pooled (+5.953, +6.729) and within
  (+8.760, +10.352) improve in both streams. Query removal hurts pooled by
  6.310 points and within by 0.684 on average (both negative in both streams).
- **Page repair fails:** within declines in both streams (-4.443, -0.293);
  pooled is mixed (-3.408, +1.356). Query removal hurts pooled in both streams
  but its within effect is mixed and +0.635 on average. The page role-asymmetry
  prediction does not replicate.
- **Bot harm survives:** removal hurts pooled and within in both streams;
  final orientation-only carries substantial harm.
- **The upstream direction-only explanation fails on political:** it hurts
  instead of repairing. Norm-only is small (+0.151 pooled, +0.087 within).
  Neither swap alone reproduces the joint improvement. This is compatible
  with a learned nonlinear operating-range interaction, not proof of its
  specific affine/normalization cause.
- **The final-contrast pooled account is not simply direction-only at 50k:**
  political orientation-only +1.373 and strength-only -0.557 do not account
  additively for joint +6.341. The factorial interaction contrast is +5.525
  AUC points; that is a contrast on this metric, not an intrinsic variance
  partition. Equal-strength political orientation changes still improve pooled
  AUC by 7.631 points and within by 9.556, ruling out pure scaling. At final
  contrast, orientation must carry within-episode ranking by construction;
  this identity alone is not a learned mechanism discovery.

Political mean contrast strength shrinks from .2835 to .0270; mean orientation
cosine is .2924. Pages shrink .2185 to .0281; bots .3850 to .0182. Near-collapsed
class references make pure upstream claims particularly unsafe here.

### Native performance and strong readouts remain visible

Mean pooled AUC; each readout sees exactly the same episodes for this checkpoint:

| Target | Native | Support removed | Raw prototype | Trained prototype | Untrained prototype |
|---|---:|---:|---:|---:|---:|
| Political | .5649 | .6283 | .7528 | .6056 | .6011 |
| Pages | .5524 | .5422 | .8496 | .5723 | .7006 |
| Bots | .6021 | .4461 | .5236 | .6199 | .6297 |

Untrained means one deterministic seed-0 initialization of the same recorded
architecture, not saved historical initial weights. Prototype temperatures
are not fitted; NLL across these readouts is not a calibrated comparison.
Political removal improves ranking but drops threshold accuracy from .4919
to .2515, despite improving NLL (.8186 to .7256). Thus even the political
result is not an all-metric performance repair or evidence of deployment value.

This test establishes effects in one much longer-trained checkpoint, but does
not establish a strong representative transfer model. Its changed training /
neighborhood recipe also prevents a duration-only causal comparison with the
2,500-step models. Same implementation and sum aggregation; no independent
implementation or public-benchmark replication was conducted.

## 5. Nine-source evidence and aggregation provenance

The existing 9-source x 5-target x 2-stream intervention map is complete
(720 original role/readout cells). The figure shows actual support-removal
deltas, not the prototype-versus-inference map. Hong Kong is an extreme, not
unique: removal helps both streams for 5/8 foreign political sources and 3/8
foreign page sources; it hurts bot AUC for all nine sources in both streams.
These are descriptive signs from distinct historical seed-0 checkpoints, not
the three-seed initial-model factorial or statistical significance claims.

Previously tested support dispersion and output sensitivity do not provide a
validated cross-target sign predictor. No new predictor search was performed.
The current tests make no unsupported claim about the other seven sources'
final-contrast mediation.

Training-time aggregation is reconstructed from recorded source revisions and
contemporaneous PyG 2.3.1 requirements: they support **sum** behavior during
training as well as this runtime. The nominated 50k model's training revision
is `0c67225c7630beb6fddcf1ad41e7fc61c968ab80`; the relevant selected SAGE
implementation is unchanged. No inference-time mean substitution is presented
as a mean-trained model. Detailed evidence remains in
`data/message_scale_training_provenance.json`.

## Deliverables, reproducibility, and boundaries

New standalone one-page argument (prior pages preserved):
`/Users/philipp/projects/gfm/paper/transfer-prediction/class-reference-decision-2026-09-07/output/pdf/class_reference_decision.pdf`.
Rendered and visually checked at 1,600 pixels, with no clipping or overlaps.
PDF SHA256: `0b7df78a14032040ac240fea165ebc6e0e2f8f6ba0250a968801e3bc16c6450f`.
The main manuscript remains SHA256
`b2f2ff5ad58f1b89a7f088baf31bf022a594325dc79305524f62165d020cbe9f`;
the first-540 and full-900 decision-page hashes are also unchanged. The PDF
skill determined the standalone builder, rendering and visual-verification
workflow; it did not trigger any manuscript rewrite.

Data: `data/classref_ranking_20260907`, `data/classref_contrast_20260907`,
`data/classref50k_20260907`; derived paired tables and hash receipts:
`data/classref_decision_20260907`. Reproduce the synthesis with
`analyze_class_reference_decision.py --output <new-directory>`; reproduce the
figure with `plot_class_reference_decision.py --output <figure.png>` under
local Python 3.11 and `MPLBACKEND=Agg`. The synthesis itself runs no models.

The final-contrast replay completed 42 paired cells / 252 score conditions,
4,032 exact stage-query checks, and bit-exact native endpoints against saved
predictions. The full 50k test completed 84 cells across 6 target-stream blocks,
3,840 exact query checks and unchanged weights. Maximum 50k native-versus-
reconstructed AUC differences are 0.000191 pooled and 0.00543 within-episode
points (rounding); no reported conclusion depends on them. Streams are not
training seeds and repeated account occurrences are not IID samples; no
significance claims or confidence intervals are invented.

Tucker worktrees / outputs:

- `prodigy-classref`, pinned `246b73da`: `log/classref_ranking_20260907` and
  `log/classref_contrast_20260907` (complete).
- `prodigy-classref50k`, pinned `22ad23ea`: `log/classref50k_20260907` (complete).
- All are under `/dataMeR1/phil/gfm/`. Heavy artifacts, sampled inputs and
  per-query tensors remain there. CPU-only diagnostics, four threads, no GPU
  allocation and no new training.

Selected weight SHA256:
`c392cb264a05ac9a21dd6d3afe4590d3a9e7e8f74079da3e2a89fe35668a6a21`.
Local worktree `/Users/philipp/projects/gfm/prodigy/.worktrees/role-topology`,
branch `codex/role-topology-interactions`. Only diagnostic source code was
transported through public Git (`93199b3d`, `246b73da`, `22ad23ea`). New results,
figures and this decision note remain local/private. The completion automation
remains paused; no new automation or experiment was scheduled. The broader
publication goal is not declared complete.
