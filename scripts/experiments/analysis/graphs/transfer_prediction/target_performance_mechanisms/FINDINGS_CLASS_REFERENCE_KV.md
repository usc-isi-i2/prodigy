# Decision: support values, not routing alone, carry the political ranking effect

**Current status:** the public test is complete and rejects the predicted
support-help/query-harm conjunction. See `FINDINGS_PUBLIC_BOUNDARY.md`.
The execution update below is historical, not a live-job status.

**Execution update, 7 September 2026:** the feasibility/status statements below
record the earlier decision point. The selected public test is now the original
Wiki-to-FB15K-237 recipe, with full Wiki training running and checkpoint 8,000
nominated before target evaluation. Its current state and frozen quality/role
decision are in [the private execution record](../../../../setup/public_prodigy_kg/README.md).
No public-target result is available yet; the scientific conclusions below
are unchanged. Do not restart the MAG-checkpoint search from this snapshot.

Follow-up accounting (no new model forwards):
`FINDINGS_CLASS_REFERENCE_ACCOUNTING.md` explains the ranking/accuracy mismatch
with actual examples and additive saved-vector terms. It rejects direct label
residual dominance while preserving the unresolved indirect label-interface
boundary. It does not establish a causal BatchNorm explanation.

7 September 2026. This advances the explanation in
`FINDINGS_CLASS_REFERENCE_DECISION.md`; it does not discard the scale, radial
swap, final-contrast, nine-source, or nominated 50k results. **Stop this control
block.** No further experiment is queued, and the main manuscript is unchanged.

## The contribution we can defend

Graph context participates in two different computations: representing queries
and constructing the class references against which those queries are scored.
The broader evidence shows that its transfer utility can differ across these
roles and reverse during training on fixed target inputs. The new result
localizes a political discrimination effect to **what support values contribute
to the class reference**, rather than requiring different attention routing.
A discovery-selected component ordering survives a change of training recipe
and checkpoint, where the earlier support-direction-only rule fails.

This is a bounded mechanism result. It is not a universal sign predictor,
proof of a particular normalization pathology, or a useful suppression method.
The evidence for role asymmetry is broader than the evidence for the K/V
explanation: K/V has been tested on **Hong Kong to covid-political only**.

## One experiment with a falsifiable ordering

The actual single-layer metagraph computes support keys and values. Its
attention depends on keys, receiving-node queries, and signed label-relation
edge attributes. Values specify the vectors being aggregated. We capture the
native intact endpoint A and support-background-message-zero endpoint B, then
transplant only the support K blocks, V blocks, or both from B into A.

- The value-only condition uses A's actual support-to-label attention exactly.
- The key-only condition uses B's support-to-label attention with A's values.
- Query and label Q/K/V rows are not transplanted; every Q block is unchanged.
- Joint K/V reproduces B's final label vectors and query logits **bit exactly**.
- Final query representations, weights, donor tensors and input batches stay
  unchanged. The original M module and final decoder are retained.

The two endpoints are support-message suppression, not deletion of any
support label or support-to-label edge. The same previous helper supplies the
alpha=0 endpoint. The model has one M layer and no final-back layer, so a
changed support residual is not consumed in a second pass. These structural
conditions are enforced, not presumed valid for arbitrary attention models.

Discovery was one existing political stream (128 episodes), the initial
Hong Kong seed-0 checkpoint at 2,500 steps. Before seeing the K/V outcomes,
the decision rule selected a singleton only if its within-episode AUC gain
was positive and exceeded the other singleton by at least one AUC point.
Ambiguous discovery would stop without forcing a prediction. This practical
margin is not a significance test.

Discovery selected **values**. The frozen prediction required value-only to
improve within-episode AUC and exceed key-only **in each** existing 50k stream.
All four inequalities passed. Pooled AUC was not substituted for the primary
metric. The same previously nominated source-validation-selected checkpoint
was used; no checkpoint search or additional training occurred.

Important chronological boundary: the 50k intact/removal endpoints, earlier
direction failure, and episodes were already known when this experiment was
designed. **Only the new K/V ordering is prospective.** This is a test of new
intervention outcomes in an existing second configuration, not a pristine
held-out dataset or an independent implementation replication.

Frozen prediction SHA256:
`fe6015c9ddeb27a821cd138022025fa862dfe07302835f5129ff4c37a4977f5d`.
The 50k protocol embeds the same prediction object. The synthesis checks
object equality and records the source-file hashes.

## Results: discrimination with attention fixed

Changes in within-episode AUC, **points**, from each native intact endpoint:

| Configuration / stream | Keys only | Values only | Joint K/V = suppression |
|---|---:|---:|---:|
| 2.5k discovery / original | +1.230 | +5.664 | +7.248 |
| 50k prediction / original | +0.152 | +6.858 | +8.764 |
| 50k prediction / fresh | +0.839 | +8.927 | +10.352 |

Each row uses 128 paired episodes, one checkpoint per configuration. The
streams are not training seeds. Repeated account occurrences are not IID;
these are descriptive estimates and a directional prediction check, not
confidence intervals or statistical significance claims. Nonadditivity
precludes treating singleton/joint ratios as fractions of mechanism explained.

Pooled margin AUC and threshold accuracy, retained together:

| Configuration / stream | Condition | Pooled AUC | Within AUC | Accuracy |
|---|---|---:|---:|---:|
| 2.5k / original | Intact | .82016 | .84107 | .72396 |
| 2.5k / original | Keys only | .83972 | .85337 | .78939 |
| 2.5k / original | Values only | .88627 | .89771 | .79232 |
| 2.5k / original | Joint | .90649 | .91356 | .80339 |
| 50k / original | Intact | .56650 | .56641 | .48698 |
| 50k / original | Keys only | .56962 | .56793 | .48340 |
| 50k / original | Values only | .60892 | .63498 | .25911 |
| 50k / original | Joint | .62603 | .65404 | .25163 |
| 50k / fresh | Intact | .56336 | .56091 | .49674 |
| 50k / fresh | Keys only | .56643 | .56930 | .47786 |
| 50k / fresh | Values only | .61082 | .65017 | .25846 |
| 50k / fresh | Joint | .63065 | .66442 | .25130 |

These use native float32 logits and double-precision logit-margin subtraction.
Tiny differences from the prior final-contrast reconstruction are rounding,
not new episode streams or changed models.

## Internal quantities: the key intervention is engaged

50k episode means over both streams (256 episodes):

| Condition | Attention TV from intact | Cosine of final contrast to intact | Raw label half-difference norm | Final contrast strength |
|---|---:|---:|---:|---:|
| Intact | .0000 | 1.0000 | 4.2740 | .28346 |
| Keys only | .1461 | .9577 | 4.1628 | .27304 |
| Values only | .0000 | .2954 | .4880 | .02384 |
| Joint | .1461 | .2924 | .5336 | .02696 |

Attention TV is the mean total variation between actual incoming attention
distributions, over the two label nodes and heads. Values-only preserves
attention exactly in every batch, not merely on average. Keys change it by
.145/.147 in the two streams, so the small key-only performance effect cannot
be dismissed as an unengaged routing intervention. At 2.5k, key-only TV is
.1604; corresponding final-contrast cosines are .9757 (keys), .8704 (values),
and .8625 (joint). The same qualitative pathway separation is visible there.

At 50k, raw label-difference cosine to intact is .9621 for keys and .3335 for
values. Thus substantial change is present **before** individual label-vector
normalization; final cosine decoding is not the only location where the
contrast changes. This is descriptive localization, not a causal partition of
common-mode, radial and tangential contributions. The value path still combines
direction, magnitude, learned affine maps and downstream normalization.

### Correction: a radial swap interaction does not prove nonlinearity

For endpoints A = r_A u_A and B = r_B u_B, direction-only is r_A u_B and
norm-only is r_B u_A. Already at the input to an affine map,

`B - direction_only - norm_only + A = (r_B - r_A) (u_B - u_A)`.

Consequently an affine downstream map can give a nonzero interaction, before
normalization or AUC is considered. The previous observation was compatible
with a nonlinear operating-range explanation but did not discriminate it from
this algebraic alternative. A synthetic test now pins that counterexample.
Do not promote the radial factorial to evidence of a learned nonlinear defect.

## The strongest counterexample is useful for the paper's identity

The 50k political effect is a discrimination improvement, **not** a deployment
repair. Saved-logit analysis gives mean within-episode AUC .56366 -> .65923,
but accuracy .49186 -> .25146. The positive class has 25% prevalence; its
prediction rate rises from 59.67% to **99.85%**. In the original/fresh streams,
123/124 of 128 suppressed episodes predict the same class for every query.
Values-only likewise yields about .259 accuracy. Positive and negative margins
can be ordered better while almost all remain on the same side of zero.

The episode-mean/within-residual decomposition reproduces native margins and
accuracy and preserves within-episode AUC after centering. It is a descriptive
analysis, not a fitted intercept or proposed adaptation method. Raw prototypes
remain stronger on political pooled AUC (.7528 vs .5649 native and .6283
suppressed). Their temperatures were not fitted; do not infer cross-readout
calibration from NLL. Near-collapsed class contrasts warrant caution.

The earlier page-repair prediction fails at 50k (-2.37 within points); bot
suppression hurts (-19.14). These refute universal suppression and weaken
generality of a particular target pattern. They do not refute the political
value-path ordering because that ordering has not been tested on those targets.

## Broader evidence and the next single generality test

The existing 9-source x 5-target x 2-stream role-intervention map is in the new
figure. Political suppression helps both streams for 5/8 foreign sources;
page suppression for 3/8; bot suppression hurts all nine sources in both streams.
These are historical single-seed checkpoints, not nine replications of the
current K/V mechanism. Hong Kong is an extreme, not the sole affected source.
Training reverses page support utility on fixed inputs: at steps 100 -> 2500,
pooled signs go from 6/6 harmful to 6/6 helpful; within signs from 5/6 harmful
to 6/6 helpful. Static target density, text similarity or collection window
cannot by themselves explain this within-training comparison.

The highest-value remaining generality test is **one original MAG-pretrained
PRODIGY checkpoint on arXiv**, preserving its actual S2/U/M protocol and first
reproducing its native accuracy. Only then should role separation be evaluated.
The official paper's relevant reference is 73.09% accuracy, 3-way/3-shot,
500 tasks; it is not binary AUC. Do not quietly substitute a binary social
episode protocol or a model without the learned M stage.

Read-only Git/Tucker and official-resource checks found loaders and public
data but no matching ready-to-use official checkpoint. Cora/PubMed graphs are
available on Tucker; they are not MAG-to-arXiv. A third-party contrastive
S2/U/A checkpoint exists but lacks the learned M module and does not meet this
test. Obtaining original weights and effective configuration is the missing
prerequisite; no large download, new MAG training, or author contact was done.

Novelty boundary: [PRODIGY](https://cs.stanford.edu/~jure/pubs/prodigy-neurips23.pdf)
already constructs label representations from supports, and
[FEAT](https://arxiv.org/abs/1812.03664) already studies support-set adaptation,
including a harmful example with fixed queries. Generic classifier construction
or "support adaptation can hurt" is not our novelty. The contribution is the
graph-context role/training interaction and the predictive localization of a
transfer effect in the actual support-value computation.
[Luo et al.](https://proceedings.mlr.press/v202/luo23e.html) also establish that
representation learning and adaptation should not be conflated.

## Reproduction and state

Code-only transport: commit `5ffbfa2b` on public branch
`codex/role-topology-interactions`. Local worktree:
`/Users/philipp/projects/gfm/prodigy/.worktrees/role-topology`.
Tucker worktree: `/dataMeR1/phil/gfm/prodigy-classrefkv`, pinned to that commit.
Complete run directories under its `log/`:

- `classrefkv_discovery_20260907`: 5 cells, 160 exact query checks.
- `classrefkv_long_20260907`: 10 cells, 1280 exact query checks.
- `class_preference_20260907`: 24 saved-score cells, no model forwards.

Both K/V runs use four CPU threads, no GPU allocation or new training. Six
synthetic tests passed locally and on Tucker; four prediction-rule cases also
passed locally. The long-model weight digest remains
`c392cb264a05ac9a21dd6d3afe4590d3a9e7e8f74079da3e2a89fe35668a6a21`.
Training-time source/runtime records support sum aggregation, not a
mean-trained model. See the prior decision note for full provenance.

Aggregate data mirrors have the same names under this leaf's `data/`.
Full activations, sampled inputs, per-query predictions and numeric episode
examples remain private on Tucker. `plot_class_reference_kv.py` produces
`figures/class_reference_kv.png` and the paired estimates, geometry and hashes
in `data/classrefkv_decision_20260907/summary.json`. It runs no models.
Use local Python 3.11 with `MPLBACKEND=Agg`; provide `--output` and
`--summary-output` explicitly. New results, figures and findings have not been
staged, committed or pushed. Existing dirty work is preserved.

The standalone one-page PDF and its builder belong in the sibling paper tree,
not the manuscript. The PDF skill supplies rendering and visual verification.
Completed output:
`/Users/philipp/projects/gfm/paper/transfer-prediction/class-reference-value-path-2026-09-07/output/pdf/class_reference_value_path.pdf`.
Rendered at 1700 pixels and inspected with no clipping or overlap; the copy
to the paper directory was hash-verified. PDF SHA256:
`4fb4f9f4d069c67d101d3214d146842a677f6d5bc7ae3a979491f9403b7347e1`.
Main manuscript SHA256 remains
`b2f2ff5ad58f1b89a7f088baf31bf022a594325dc79305524f62165d020cbe9f`;
the previous class-reference and complete-scale pages retain their recorded
hashes. Fresh Git/Tucker checks agree on `5ffbfa2b`; the experiment worktree
is clean remotely, no user Python/tmux jobs remain, and GPUs are idle.
Previous pages and the main draft are retained. No new automation was created;
the completion automation remains paused. The broader publication goal is
active, not declared achieved merely because this control block is complete.
