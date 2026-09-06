# Mechanism results: target signal, learned readout, and a failed coverage prediction

2026-09-06 UTC. The historical nine-source analyses use training seed 0; the
completed controlled retraining uses three seeds. This is not a complete causal
source-ranking explanation or a submission-ready ICLR result. All five targets,
both episode streams, all 24 controlled models, the planned cue analysis, and an
exact-initialization supplementary reference are complete. The prespecified
coverage prediction is not supported: its primary effect reverses across seeds.
The subsequent 18-model readout-training intervention is also complete. Its
Facebook full-model consistency prediction fails despite positive readout-probe
effects in every pair on both streams. A separate same-seed control audit finds
substantial cross-launch drift, documented below rather than called replication.
A bounded numerical audit now localizes a sufficient source of CPU update drift
to the cosine decoder's repeated advanced indexing, on fixed synthetic inputs
and hash-matched real training inputs with the actual initial weights. Its
historical target-AUC contribution remains unquantified.

## What changed our next experiment

The evidence does not support one universal explanation such as graph size or
biography distance. It identifies two distinct bottlenecks worth manipulating:

1. **Target-dependent use of available signal.** Graph context nearly solves
   Election with no graph-model training; Facebook has much stronger center-only
   signal that current pretrained models do not exploit as well. COVID Political
   benefits substantially from learned neighborhood representations.
2. **Source-dependent quality of the sampled NM task.** The actual member selector
   concentrates some sources on a small, repeatedly reused set of nodes. Within
   a 30-way episode, the same node can receive conflicting pseudo-class labels.
   This is far more severe for Hong Kong and COVID Political than for COVID,
   Ukraine, or TwiBot. A fixed-walk member-selection control reduces it.

Neither result alone establishes why changing a training source causes final
target performance. The controlled member-selection experiment below successfully
changes exposure but does not produce the predicted robust transfer benefit.
The stage-resolved target-signal mismatch survives this failed explanation.

## Exact comparison contract

Nine specialists plus one deterministic random initialization, identical cached
128-episode 2-way/10-shot test inputs within each target. Probes use only each
episode's support labels. Ridge regularization is fixed at 1 after row-wise
feature normalization; no query labels select it. No model weights are updated.

**Label-interface clarification:** these original-feature graph runs do not load
a sentence encoder for labels unless one is explicitly requested. With the
current empty `label_emb_model`, classification uses 768-dimensional standard-
normal vectors generated deterministically from each class index/name, then
projects them. They are **not semantic label-text embeddings**. NM training
instead uses its frozen 256-dimensional label table. The table and initial label
projection have identical tensors across all 36 historical checkpoints (all nine
sources, four steps). Historical control names such as `zero_label_text` are
retained for artifact compatibility but mean erasing those input label vectors.

Facebook's label universe is the 30 page categories, with two categories sampled
per episode; its raw text is a page description, not a Twitter biography. The
production metric uses episode-local binary labels for this episodic label
space, while the other four targets use their fixed global binary mapping.
The comparison preserves that existing metric contract for every decoder.
Facebook's raw-center advantage is also present in decision accuracy (Ukraine
full model .6924 versus raw-center ridge .8398 on the original stream), so it
is not only a pooled-AUC calibration difference.

For all 45 pretrained baseline cells, reference episode hashes match and the
largest aggregate metric difference from historical GPU evaluation is 1.70e-6.
Trace/no-trace logits are bit-identical on the same device. The cached-batch hash
also covers features and all metagraph tensors. Historical fingerprints already
included sampled context IDs and edges, not just centers/labels.

The first TwiBot cell stopped the larger sweep: CPU/GPU AUC differed by 1.70e-6,
while accuracy/F1 and episode hash matched. Its two-target continuation explicitly
allows at most 1e-5 AUC portability error, keeps decision checks at 1e-6, and
records observed errors. The failed partial run is preserved. The continuation
is complete, with its own `DONE` marker; combined, the two runs cover all five
targets. The summary has 1,250 diagnostic rows.

## Target inputs versus model use

Same-query AUC, with the same support-only ridge decoder for the first two columns:

| Target | Raw center features | Raw neighborhood mean | Best full specialist |
|---|---:|---:|---:|
| COVID Political | .8231 | .8629 | .9464 (Ukraine) |
| Facebook Page Reference | .9196 | .7531 | .7738 (TwiBot) |
| Election2020 | .8401 | .9868 | .9909 (TwiBot) |
| TwiBot20 | .5531 | .5772 | .6356 (Ukraine) |
| Ukraine Suspended | .5131 | .5211 | .5236 (COVID Political) |

This directly distinguishes readily decodable input signal from final PRODIGY
performance. Facebook's poor result cannot simply be blamed on absent signal in
the supplied inputs. Conversely, Election's near-ceiling results largely reflect
label information available in sampled neighborhood features before NM training.
These are 10-shot episodic probes, not fully supervised references or Bayes limits.

### TwiBot exposes a different bottleneck

An untrained encoder's pre-metagraph representation plus support-only ridge
reaches .7114 on TwiBot, above every full specialist. A separate probe using only
five sampled-topology summaries reaches .7040; raw center features plus these
summaries reaches .7096, while a fixed 256-dimensional random projection of the
center features remains at .5541. Node count alone is only .5314. This narrows the
lead to structural information beyond sampled size, rather than simply a random
projection of biography semantics. It does not yet establish which structural
property the untrained encoder uses or why NM training fails to exploit it.

Individual-scalar probes identify an even stronger TwiBot signal: sampled center
in-degree alone reaches .7238. This is incoming degree in our **reconstructed
retweet graph**, not follower count in the original TwiBot follow graph. The
builder uses retweeter-to-retweeted edges; degree refers to the sampled input
subgraph, not full-network popularity. Its construction/collection dependence
must be considered before claiming a general bot-detection mechanism.
The binary presence of any incoming sampled edge already yields .6858/.6730 on
the original/fresh streams, versus .7238/.7112 for the count. This narrows the
lead further toward being observed as retweeted, without accounting for the
entire degree association.

Election is different again: center norm, context-mean norm, and center/context
cosine together reach .9843, while topology alone is .4937. Context-mean norm
alone reaches .9862, and coherence (that norm divided by mean context-vector
norm) reaches .9863. Context nonzero fraction gives .5000 and center norm .4968.
The artifact's embedding metadata reports zero empty bios among all 78,932
rows. Missing bios do not explain this result. A high neighborhood-vector norm
indicates more aligned feature directions, but does not by itself identify why
one class has that alignment. Compact evidence: `data/input_scalar_probes.json`.
The follow-up computes mean off-diagonal pairwise cosine among nonzero,
unit-normalized context vectors, removing the diagonal `1/n` contribution to
the squared mean norm. It gives .9863/.9760 on original/fresh episodes. Thus the
cue is not just that algebraic finite-neighborhood-size term. This is not a full
statistical or causal adjustment for collection density. The follow-up evidence
is in `data/cue_controls_original.json` and `data/cue_controls_fresh.json`.

### Fresh episodes replicate the bottlenecks, not every donor ranking

The same frozen checkpoints were replayed on offset **100003**, with new episode
fingerprints on every target. All probes again fit only episode supports.

| Diagnostic AUC | Original episodes | Fresh episodes |
|---|---:|---:|
| Facebook raw center + ridge | .9196 | .9301 |
| Facebook best full specialist | .7738 | .7680 |
| TwiBot sampled center in-degree | .7238 | .7112 |
| TwiBot untrained pre-metagraph + ridge | .7114 | .6953 |
| TwiBot best full specialist | .6356 | .6417 |
| Election neighborhood coherence | .9863 | .9767 |
| Election best full specialist | .9909 | .9872 |

TwiBot's best full donor changes from Ukraine to Election2020 on the fresh stream;
the Ukraine/TwiBot singleton Facebook ordering also changes. Do not present those
rankings as stable conclusions. These are two episode streams from the same
domains and frozen training seed, not independent training or domain replications.

### Training trajectories distinguish underuse from deterioration

Stage names refer to **observation points**, not three independently learned
modules. `S0_pool` is the mean of post-convolution features before the encoder's
learned center/mean readout. `U1_pre_meta` is the resulting learned center
readout passed to the metagraph. The default U operation itself is parameterless
and copies that center to its supernode; the learned readout weights live in S.
Thus a declining U-stage probe does not identify a faulty learned U module.

All nine historical seed-0 specialists are replayed at steps 100, 300, 900, and
2,500 on both cached episode streams. This is a complete 360-cell full-model
grid (6,120 diagnostic rows), not target-selected checkpoints. All 36 checkpoint
digests match between streams; terminal checkpoint metrics match the prior
replays, and every cached input tensor hash matches the established reference.
There is no saved historical step-0 checkpoint: the separate random encoder is
**not** a verified matched initialization for these trajectories.

Mean AUC change from step 100 to 2,500 across the nine source models:

| Target / stage | Original episodes | Fresh episodes |
|---|---:|---:|
| COVID Political: pooled S + ridge | +.2285 | +.2146 |
| COVID Political: full model | +.2130 | +.2049 |
| Facebook: pooled S + ridge | +.2188 | +.2335 |
| Facebook: learned U + ridge | +.2210 | +.2225 |
| Facebook: full model | +.1681 | +.1691 |
| TwiBot: post-convolution center + ridge | −.0211 | −.0372 |
| TwiBot: pooled S + ridge | +.0472 | +.0496 |
| TwiBot: learned U + ridge | −.0273 | −.0365 |
| TwiBot: full model | −.0779 | −.0809 |

Facebook improves for **all nine sources at all three stages on both streams**,
but still falls short of the raw-center probe. Thus its current deficit is not
evidence of progressive deterioration between steps 100 and 2,500. The input
signal is available but inadequately used by this trained architecture/protocol.

TwiBot shows a different trajectory. Pooled-S decodability improves for eight of
nine sources on each stream, while full-model AUC declines for eight of nine.
Learned-U decodability declines for eight sources on the original stream and all
nine on fresh episodes. **Ukraine is the only source whose full-model TwiBot AUC
improves on both streams** (+.0513/+.0397). The post-convolution **center branch
already declines for all nine sources on both streams**, before the learned
readout mixes it with the improving pooled branch. This is a branch-dependent
training-associated mismatch, not proof that the readout weights caused it, a
causal decomposition of stage AUCs, or information destruction. Intermediate
curves can be nonmonotonic and are all retained, not used for model selection.

Election also improves strongly (mean full-model changes +.3559/+.3622), while
suspension remains near its current bio-only floor (−.0037/−.0031). These remain
one training seed on five datasets. Evidence: `data/trajectory_cells.csv`,
`data/trajectory_endpoint_summary.csv`, `data/trajectory_validation.json`, and
`data/trajectory_input_validation.json`; figure `figures/training_stage_changes.png`.

**Post-hoc cue agreement narrows the TwiBot lead.** On those same saved query
logits, compute mean within-episode Spearman correlation between each model's
binary margins and three fixed support-only probe margins: sampled center
in-degree, raw center features, and raw context features. This fits no query
labels and avoids pooling incompatible episode-local class orientations. All
128 episodes have nonconstant scores for every comparison; all 648 cells match
the independently validated checkpoint and input provenance.

From step 100 to 2,500, learned-readout agreement with the in-degree cue declines
for **all nine sources on both streams** (mean changes −.2605/−.2778), while its
agreement with both raw-center and raw-context predictions increases for every
source. Full-model in-degree agreement declines for eight sources on both streams
(mean −.4308/−.4674). Ukraine is again the only increasing source
(+.0927/+.0628), matching its exceptional positive full-model AUC trajectory.
Pooled-S in-degree agreement has mixed signs, unlike the consistent later-stage
decline. This is consistent with a training-associated shift in cue use, not
simply every encoder feature becoming worse. It does **not** prove that degree
is the sole causal mediator, that information was destroyed, or that increasing
cue agreement by intervention would improve generalization. Evidence:
`data/twibot_cue_alignment_{cells,changes,summary}.csv` and
`data/twibot_cue_alignment_validation.json`.

### Suspension: test the input restriction, not just another source mixture

The source table contains account metadata omitted from the shared 768-dimensional
bio input. A fixed 12-feature probe (log-counts/account age and verified status),
scaled using only episode supports, reaches **.6730/.7141** on the original/fresh
streams. Raw biography ridge reaches .5131/.5248 and the best full specialists
.5236/.5402. Metadata missingness alone gives .5000 on both streams.

All 72,295 raw-row labels match the graph artifact; every cached center feature
matches the same original graph row exactly, and local episode labels match the
global row labels. Thus this is not an unchecked cross-table join. Nevertheless,
the metadata were **not inputs to the pretrained models**, and their timing
relative to suspension is unverified. This demonstrates usable signal in an
omitted modality, not a like-for-like model gain or a validated prospective
suspension detector. The current floor must not be treated as an intrinsic task
limit. Evidence: `data/metadata_probe.json`, `data/metadata_probe_protocol.json`.

The upstream study also reports much stronger suspension prediction with account
metadata and tweets than with profiles alone, under a different supervision
protocol. Its scores are not direct baselines for our episodes.
[Social-LLM, Table 2](https://arxiv.org/pdf/2401.00893).

Election/COVID Political labels themselves were derived using profile hashtags
and media-endorsement heuristics. This is relevant to why these inputs are
predictive, but does not alone establish accidental train/test leakage.
[Retweet-BERT, pseudo-label generation](https://arxiv.org/pdf/2207.08349).

The raw-plus-topology probes use equally weighted normalized feature channels,
fixed regularization, and support-only scaling. Their poor scores on some targets
do not show that adding information is harmful in principle; this is one fixed
decoder, not a tuned optimum.

### The Ukraine–TwiBot contrast is stage-dependent

| Target / stage | Ukraine | TwiBot | Ukraine minus TwiBot |
|---|---:|---:|---:|
| COVID Political: S pooled features + ridge | .9268 | .9269 | −.0001 |
| COVID Political: learned readout + ridge | .9444 | .9373 | +.0071 |
| COVID Political: full model | .9464 | .9185 | +.0279 |
| Facebook: S pooled features + ridge | .8484 | .8704 | −.0220 |
| Facebook: learned readout + ridge | .8048 | .8232 | −.0184 |
| Facebook: full model | .7720 | .7738 | −.0018 |

The singleton Facebook difference is tiny. Matched-pair replay is now complete
for 14 models on both targets, with all 28 baseline reference parity checks
passing. Excluding pairs containing the target leaves six shared foreign
partners per target. Mean Ukraine-minus-TwiBot AUC across these matched partners:

| Target | S pooled + ridge | Learned readout + ridge | Full model |
|---|---:|---:|---:|
| COVID Political | +.0124 (6/6 positive) | +.0084 (5/6) | +.0172 (6/6) |
| Facebook | −.0131 (6/6 negative) | −.0261 (6/6) | −.0296 (6/6) |

On the original episodes, the pair reversal is already visible in the learned
pooled encoder features under a common support-only decoder. It is **not only
a final-metagraph effect on that stream**.
For Facebook, it grows through the learned readout. The six partners are six
composition contrasts, not six independent training-seed replications. See
`data/pair_stage_contrasts.csv` and `data/pair_validation.json`.

**Fresh-pair replication qualifies the reversal.** With unchanged checkpoint
hashes and new target episode fingerprints, COVID Political keeps a +.0144 full
model advantage and +.0081 pooled-feature advantage for Ukraine, both positive
for all six partners. Facebook's mean full-model contrast is still negative
(−.0107), but only four of six partners favor TwiBot; pooled features average
−.0032 with three of six favoring it. The readout-stage mean is −.0085 (five of
six negative). Thus the universal Facebook sign and every-partner encoder
reversal **do not replicate**. Among the five partners foreign to both targets,
the source-by-target interaction has the original direction for four of five
on fresh episodes. Do not pool the two targets' six-partner averages as though
their partner sets were identical. Evidence: `data/pair_stage_replication.csv`
and `data/fresh_pair_validation.json`.

On COVID Political, the larger final donor gap is not present with the common
decoder on pooled S features. On Facebook, TwiBot's advantage is visible before
the learned readout, but the full decoder largely erases it. This locates useful
diagnostic boundaries; subtracting probe AUCs is **not** a causal decomposition.
Lower linear/ridge decodability also does not prove information-theoretic loss.

### Neighborhood membership matters more than these message-passing edges

On COVID Political, shuffling only non-center feature rows within an episode
drops Ukraine from .9464 to .6765 and TwiBot from .9185 to .6263. Center features,
graph edges, sampled sizes, and the episode feature multiset are preserved.
Removing S background edges instead gives .9497 and .9355 respectively.

This does **not** establish that graphs are unnecessary: the sampled neighborhood
membership and mean-pooling context still come from the graph. It separates the
information supplied by graph-selected context from explicit message passing on
those sampled edges. On Facebook, context shuffling also hurts the trained
models, despite a better raw center-only decoder; a trained model can depend on
context without being the best way to use the available inputs.

Fresh-episode interventions replicate a sharp task-dependent distinction for
the **same Ukraine-trained weights**:

| Target | Shuffle context features: ΔAUC original / fresh | Remove background edges: ΔAUC original / fresh |
|---|---:|---:|
| COVID Political | −.2699 / −.2646 | +.0033 / +.0002 |
| Election | −.3484 / −.3929 | −.0004 / −.0027 |
| Facebook | −.1449 / −.1075 | −.0048 / +.0025 |
| TwiBot | −.0148 / −.0027 | −.0637 / −.0646 |

On the political tasks, graph-selected neighborhood **features** matter much
more than the remaining message-passing edges; on TwiBot, edge-dependent
information is important and permuting context features has little effect.
This is a sensitivity result for fixed sampled inputs, not a proof that one
particular scalar is the model's sole mechanism. All 60 fresh intervention
cells match baseline checkpoint hashes and original cached input fingerprints
for that fresh stream. Data: `data/fresh_intervention_deltas.csv`.

Zeroing input label vectors barely changes Ukraine/TwiBot on these targets.
Erasing support-label relations gives much worse, roughly chance-level results
on COVID Political and Facebook, but **not universally chance**: Election retains
strong positive or inverted rankings with class-keyed random label vectors still
present. This must not be attributed to label-text semantics. A joint erasure of
support relations and input label vectors gives exactly .5000 AUC in
all 15 fresh cells (three donor models × five targets). A unit test also changes
query labels alone and verifies bit-identical model logits. This is a useful
sanity check on prompt-label information, not a complete dataset-leakage audit.
Target-batch BN
moments do not close the Facebook gap. Those BN tests use query covariates and
are explicitly transductive, not valid leakage-free adaptation claims.

## Is the target simply farther away in biography-feature space?

For each source and each of two member-selection policies, sample 4,096 reference
draws from the source-audit features. Compare uniform sampling of unique members
with sampling proportional to their observed exposure counts (with replacement).
The latter intentionally has fewer unique references for concentrated sources.
Compute nearest cosine similarity for every cached query's **center biography**.
Exclude zero query vectors from semantic-similarity summaries and exclude the
target's own source model from foreign-transfer analyses. No target labels enter
the reference selection or distance. The audit is simulated training exposure,
not the recovered historical realized training stream.

Across eight foreign donors, mean proximity has positive but imperfect rank
association with AUC: on the production/uniform-unique reference it is .31 for
COVID Political, .43 for Election, .62 for Facebook, .29 for TwiBot, and .19 for
Suspension. These are eight-point descriptive correlations, not causal evidence
and not independent training replications.

But the more discriminating query-level test is nearly flat. Within each target,
double-center the source-by-query similarity and correctness matrices: remove
each source's mean quality and each query's shared difficulty. Their residual
correlation has magnitude below .036 in **all 20** target × policy × reference
settings (below .015 for COVID Political, Facebook, and TwiBot). NLL gives the
same qualitative result. Repeated queries/episodes are not treated as independent
observations, and no significance test is reported.

Thus proximity can weakly rank broadly useful sources without explaining which
model handles which query. This result weakens a simple nearest-training-bio
explanation; it does **not** rule out neighborhood-distribution differences,
conditional task alignment, nonlinear effects, or different distance metrics.
Evidence: `data/source_coverage_cells.json`, `data/source_coverage_associations.json`,
and `data/source_coverage_cell_correlations.csv`.

## The source sampler creates sharply different training tasks

256 newly simulated 30-way/3-shot/4-query episodes per source: 53,760 member
positions each. The exact stored merged `static_train` graph, production graph
builder, random walker, eligible-anchor filter, and member selector are used.
This samples the historical **distribution**, not the unavailable historical
multiworker realization. No training checkpoints were changed.

| Source | Eligible anchor fraction | Repeated positions within an episode | Effective member count across audit |
|---|---:|---:|---:|
| COVID | 11.7% | 1.6% | 5,192 |
| Ukraine | 14.5% | 3.4% | 2,448 |
| TwiBot | 48.9% | 3.1% | 2,591 |
| Midterm | 7.4% | 10.6% | 588 |
| COVID Political | 6.9% | 37.8% | 109 |
| Hong Kong | 6.3% | 39.0% | 75 |
| Election2020 | 84.7% | 24.3% | 214 |
| Ukraine Suspended | 22.6% | 11.4% | 482 |
| Facebook | 4.3% | 5.1% | 1,238 |

Repeated-position fraction is `1 − distinct_member_ids / 210`, averaged over
episodes. The selector has seven unique members per pseudo-class, so these
repetitions occur across different pseudo-classes. Different sampled contexts
can accompany a repeated center; the table is not an irreducible-error bound.
Effective member count is `(sum counts)^2 / sum(counts^2)`, not distinct-node
count and not statistical effective sample size for a confidence interval.
Eligible fraction refers to the degree prefilter, not successful unique-member
feasibility: duplicate adjacency entries can pass that prefilter but fail the
subsequent seven-unique-member rejection check.

The production selector takes the seven smallest IDs among unique random-walk
endpoints and assigns the first three to support. Holding successful anchors and
walk endpoints fixed but sampling seven unique endpoints uniformly reduces
within-episode repetition on every source. For Hong Kong it falls from 39.0% to
29.5%, and effective member count rises from 75 to 150. COVID rises from 5,192 to
10,953 effective members. Randomizing roles alone retains the original frequency
distribution, providing a separate ordering control.

This is a concrete way graph structure and construction order can act through
the training sampler. It also explains why raw node count is not enough to
describe training exposure. Yet simple raw-feature NM task ease is not a universal
transfer predictor: Facebook has the best raw NM prototype accuracy (.315), but
is a poor downstream donor. Keep conditional task alignment and learned use in
the explanation; do not rename the project around sample coverage yet.

### A graph-renumbering dependence, not a graph-similarity property

Condition on a successful random walk with a fixed set of unique endpoints.
The original rule retains the smallest numerical IDs and sorts support before
query. A pure renaming of those same nodes can therefore change both the retained
identities and their roles, without changing graph structure or node features.
Shuffling roles alone removes neither the retention bias nor its dependence on
node naming. Uniform retention fixes the unordered member-set law, but retaining
sorted roles still depends on the names. Only **uniform retention plus shuffled
roles** makes the ordered member-selection law invariant in distribution under
arbitrary renaming, conditional on that endpoint set.

For `m` unique endpoints and `k` retained members, the last policy gives each
ordered distinct `k`-tuple probability `1 / (binom(m,k) * k!)`. This is a
conditional sampling statement, not pathwise equality for a fixed RNG seed or
a verified invariance theorem for every other part of the training pipeline.
Four exact unit tests exercise the actual selection method; the small exhaustive
case checks all 24 renamings and all equiprobable retention/role permutations.
The historical rule changes its selected original identities under reversal.
Tests: `data/tests/test_member_permutation.py` (four passed).

This establishes an implementation-level dependency that ordinary isomorphism-
invariant graph similarities cannot describe. It does **not** establish how much
that dependency changes target performance. The running factorial experiment
tests policy effects; it is not a separate fixed-global-renumbering experiment,
and uniform draws on each episode are not the same joint process as applying one
fixed random node-ID permutation to the whole graph.

## A targeted architecture hypothesis that did not explain the gap

The metagraph projects each attention-weighted message with an affine map, then
sums. Its output-projection bias is therefore added once per incoming edge, not
once per destination. An analytic unit test proves this term. Default example
degrees change from 31 in 30-way NM to 3 in binary CLS; label-node degrees change
from 91 to 21 with the current support counts.

A diagnostic removed only that excess bias, holding all attention coefficients
and learned weights fixed. It completed all 45 specialist-by-target cells with
matching reference episode fingerprints. The results are nearly unchanged;
the exact paired deltas are in `data/meta_bias_deltas.csv`. On the COVID Political
target, the Ukraine-trained donor changes from .9464 to .9465 and the
TwiBot-trained donor from .9185 to .9179 (these are not TwiBot-target scores).
This control does not explain the large observed transfer gaps. Do not prioritize
a bias-placement training intervention on the strength of the implementation
detail alone. The negative result is useful: it removes one plausible distraction.

## Label-interface mismatch does not explain the target deficits

The complete frozen-weight diagnostic contains nine specialists × five targets ×
six conditions × two episode streams (540 full-model cells; 1,980 diagnostic
rows). Every baseline reproduces its previous metrics; all 320 cached batch
hashes match the references, checkpoint hashes are unchanged, and all scoped
norm/flag audits pass. The conditions were fixed before their outcomes.

Restoring the frozen training label table changes mean TwiBot AUC by only
**+.00004 / −.00023** (original / fresh); matching projected-vector magnitude to
the training table gives **+.00046 / +.00035**. Permuting the active training-table
slots gives −.00034 / −.00086. Even the largest individual TwiBot improvement
among these declared controls is below .005 AUC. Zeroing the entire projected
label input, including its bias, gives mean −.00023 / −.00026.

Facebook's corresponding training-table effect is −.00043 / −.00062; scale
matching is −.00049 / −.00046. Neither comes close to the approximately .15–.16
gap from the raw-center support-only probe. Across every target and condition,
the largest absolute source-mean change is .00298; the largest individual change
is .01349, on the low-performing suspension task under the fixed table permutation.
These are paired diagnostic effects, not confidence intervals or independent
training-seed replications. Full per-source outcomes, including negative changes,
are preserved in `data/label_interface_deltas.csv`.

Thus these particular train/test label-interface differences do **not** account
for the large target deficits. This does not prove every possible metagraph
interface change is irrelevant. It does rule out prioritizing this simple
interface restoration as the explanation or a demonstrated rescue. Support-label
relations remain essential; invariance to label-vector changes is not invariance
to erasing the support examples' labels.

## Existing NM interventions do not supply a general classification remedy

The complete compatible campaign extension reuses 15 eight-source training runs,
all excluding TwiBot. Thirty declared checkpoints keep **common 6,000 updates**
and **original source-validation-selected** comparisons separate. Both streams,
all five classification targets, and all 17 decoders are complete: 5,100 rows,
300 full-model cells. All 320 cached batch hashes match prior references; all
30 strict finite checkpoint/forward checks and weight digests pass. The selected
checkpoint file hashes also match the original NM result files exactly.

This is one training seed, batch size one, learning rate .001, and a maximum
10,000-update campaign, not the historical specialist protocol. Only within-
campaign contrasts share that recipe. Of the original 18 endpoint models, three
requiring feature-standardized/source-affine/wider forward implementations are
explicitly excluded before classification outcomes. The budget endpoint is a
baseline-distribution repeat; the selected recipe repeats the objective arm with
the same training seed. Neither is an extra seed or an independent intervention.

**At the common budget, no alternative has a positive TwiBot classification
change on both streams.** Twelve of fourteen are negative on both; auxiliary
reconstruction and the budget repeat change sign across streams. Baseline AUC is
.6663/.6367. The largest declines include low-degree eligibility
(−.0949/−.0750), blocked scheduling (−.0863/−.0742), and degree-balanced centers
(−.0745/−.0413). Uniform-neighbor positives give −.0492/−.0402. Those positives
draw directly from unique adjacency neighbors: they are not the new factorial's
uniform retention from matched random-walk endpoint sets.

**NM gains do not guarantee classification gains on the same held-out graph.**
The following comparisons use each run's originally selected checkpoint for
both tasks, including an 8,000-update baseline. They do not attach selected NM
metrics to the common-6,000 classification checkpoint:

| Intervention | Selected TwiBot NM ΔAUC | Selected CLS ΔAUC, original / fresh |
|---|---:|---:|
| Proportional source exposure | +.00749 | −.01810 / −.02785 |
| Uniform-neighbor positives | +.00408 | −.00314 / −.01190 |
| Degree-matched competing classes | +.00323 | −.00948 / −.00018 |
| Low-degree eligibility | −.00704 | +.02303 / +.01987 |
| Auxiliary reconstruction | −.00019 | +.01011 / +.01416 |
| Objective-only recipe repeat | +.00019 | −.03305 / −.03417 |

The last two NM changes are near zero, not substantive NM wins. The recipe's
opposite classification outcome makes the modest auxiliary gain particularly
unsafe to present as a reliable method. Low-degree eligibility's selected
checkpoint is update 2,000, versus baseline 8,000; its positive selected-workflow
effect reverses at common update 6,000. That is sensitivity to the full training/
selection workflow, not an isolated eligibility benefit or proof that earlier
stopping alone causes the difference. No checkpoint or variant was selected
using these classification outcomes.

Stage comparisons again resist a single "better representation" story. At common
6,000 updates, uniform-neighbor positives improve pooled-S TwiBot ridge AUC by
.0098/.0059, while the learned-readout probe declines by .0209/.0326 and the
full model by .0492/.0402. Gradient normalization improves both pooled-S and
readout probe AUC on both streams but reduces the full-model result. These
matched-decoder diagnostics are not causal mediation or additive decompositions.
All target/stage outcomes remain in the exported tables, including improvements
on other targets. The proper conclusion is **no robust remedy established by
this one-seed compatible reuse**, not that every possible data intervention fails.

Evidence: `data/campaign_cls_{cells,deltas}.csv`,
`data/campaign_cls_validation.json`, `data/campaign_nm_cls_selected_deltas.csv`,
`data/campaign_original_selected_nm.csv`, and
`figures/campaign_nm_cls_comparison.png`. Four analysis unit tests pass, including
missing-grid, changed-input/checkpoint, raw-probe, and checkpoint-rule guards.

## Completed three-seed factorial: coverage is not a sufficient explanation

Twenty-four matched models cross two sources (Ukraine/Hong Kong), three training
seeds, lowest-ID/uniform endpoint retention, and sorted/shuffled support-query
roles. Each receives 2,500 updates and 10,000 episodes. All 120 planned checkpoint
files are present. Within each source/seed, initial weights, consumed anchors and
final walk-RNG state match across policies; role pairs retain exactly the same
member sets. Initial weights also match across sources within seed. Independent
effective-config and consumed-source-label audits pass. Every terminal weight is
finite and changed, and all 320 cached evaluation-batch hashes match the prior
original/fresh references. This is 240 full-model cells / 4,080 diagnostic rows.

**The manipulation worked.** Hong Kong repeated positions fall from 39.51% to
29.80% and effective member count rises from 74.63 to 148.62. Ukraine changes
from 3.54% to 2.24% and 2,516 to 4,006. These are actual consumed training
episodes, not the earlier simulated audit. The policies do not equalize the
sources; changed identities also alter feature/context coverage, so this is not
an isolated repetition-only intervention.

The prespecified primary endpoint averages retention effects over both role
orders and the fixed COVID Political/Facebook/TwiBot panel. Positive values in
the last column mean a larger Hong Kong effect, not necessarily an absolute gain:

| Seed | Hong Kong ΔAUC, original / fresh | Ukraine ΔAUC, original / fresh | Hong Kong minus Ukraine, original / fresh |
|---|---:|---:|---:|
| 0 | −.01997 / −.01541 | +.00035 / +.00259 | −.02032 / −.01800 |
| 1 | +.02372 / +.01732 | +.00289 / +.00154 | +.02082 / +.01578 |
| 2 | −.01664 / −.01120 | +.00492 / −.00057 | −.02156 / −.01063 |
| Descriptive mean | −.00430 / −.00310 | +.00272 / +.00118 | **−.00702 / −.00428** |

Both episode streams agree on the primary sign within each seed, but seeds
disagree. **The predicted stable Hong Kong benefit is not established.** This
does not refute every sampler effect or rule out benefits in other settings.
All-target effects, role main effects, interactions, and all three seeds remain
in `data/member_factorial_contrasts.csv`; no target/checkpoint is selected.

Secondary full-model means illustrate the heterogeneity. Hong Kong retention
changes COVID Political by −.02184/−.01572, Election by −.02619/−.02059, Facebook
by +.00078/−.00958, and TwiBot by +.00817/+.01601. Its TwiBot gain is not shared
by all seeds on both streams (seed 0 reverses −.01032/+.00677). Ukraine's role
shuffle averages +.02019/+.00982 on Facebook and +.00992/+.01117 on TwiBot, but
neither benefit has a positive effect for every seed on both streams. Hong Kong
also has sizeable retention-by-role interactions: COVID Political's mean is
−.08322/−.08168. A factorial main-effect average must not be read as an additive,
policy-independent mechanism.

**Cue use changed, but that does not rescue the prediction.** The planned
secondary analysis contains all 432 model/stage/cue/stream cells and matches
the validated primary weights and inputs. Hong Kong retention increases mean
learned-readout agreement with the support-only incoming-degree probe by
.07405/.08145; full-model agreement increases .03757/.07755. Ukraine full-model
degree agreement rises .03852/.04059, positive for all three seeds in both
streams, yet its TwiBot retention AUC effect is not positive in all those cells.
Thus greater agreement with this useful cue is not sufficient for a universal
performance gain. These rank agreements are not causal mediation estimates.
All correlations here have 128 valid nonconstant episodes; the analysis retains
undefined values rather than silently dropping any condition if that changes.

Evidence: `data/member_{replay_cells,factorial_contrasts,primary_contrast,consumed_exposure}.csv`,
`data/member_intervention_validation.json`, `data/member_training_contract_validation.json`,
`data/member_cue_alignment_{cells,contrasts}.csv`, and
`figures/member_policy_primary.png`.

## Exact starting weights qualify what "training hurts TwiBot" means

An explicitly exploratory supplement was recorded at 10:05 UTC, after training
but before calculating complete factorial effects; the primary analysis was run
first. The three saved step-zero states (not freshly sampled random networks)
are each shared by all eight source/policy models in that seed. Replaying them
produces 510 diagnostic rows; exact weight digests and all 320 input hashes match.
Every one of the 4,080 terminal cells is paired to its own true initialization.
This does not recover the historical nine-source experiment's missing step zero.

For the original production policy (lowest-ID members, sorted roles), mean
TwiBot AUC changes from step 0 to 2,500 are:

| Source | Center-branch ridge, original / fresh | Pooled-S ridge, original / fresh | Learned-readout ridge, original / fresh | Full model, original / fresh |
|---|---:|---:|---:|---:|
| Hong Kong | −.07163 / −.07562 | +.06597 / +.07366 | −.01372 / −.02208 | +.10783 / +.10695 |
| Ukraine | −.03199 / −.04185 | +.04688 / +.05086 | −.02857 / −.03447 | +.08947 / +.07162 |

All six standard-policy models improve at pooled S and decline at both the
post-convolution center branch and learned readout, on both streams. This
readout decline is also present for **all twelve
Ukraine policy/seed models** on both streams, while every one of the 24 models
improves at pooled S. Hong Kong alternatives can partly reverse its readout
decline: uniform retention plus shuffled roles has mean +.01381/+.01324 versus
initialization, but one fresh-seed change is slightly negative (−.00056).

**Full-model training is not universally harmful relative to initialization.**
For each source, seeds 0 and 2 improve full-model TwiBot AUC and seed 1 declines,
under all four policies and both streams. Initial full-model AUC itself varies
greatly across seeds: .3305/.3330, .6913/.6823, .5813/.5683 (original/fresh).
The initial learned-readout ridge is far more stable: .7110–.7115 original and
.6945–.6961 fresh. A single arbitrary random full model is therefore a poor
reference for a claim about the effect of training. The historical step-100 to
2,500 deterioration and the new step-zero comparison answer different questions.

The center-branch audit materially qualifies the earlier pooled-S/readout
contrast: the readout receives one branch whose TwiBot decodability has already
declined and another whose decodability improved. These are parallel inputs,
not successive layers in a simple information-loss chain. We cannot locate the
cause in readout weights without intervention. U itself is parameterless, and
different-stage probe AUCs are not an additive causal decomposition. We still
need an independently validated intervention that preserves useful target cues
and predicts downstream performance without looking at query labels.

Evidence: `data/member_initial_cells.csv`, `data/member_initial_to_terminal_{changes,summary}.csv`,
`data/member_initial_validation.json`, and `data/member_initial_input_validation.json`.

## Directional readout-weight diagnostic: complete, source-dependent effects

The exploratory follow-up recorded at 10:16 UTC replaces only
`layer_list.0.reset_mlp_{c,m}.{weight,bias}` between each model's exact initial
and terminal states, in both directions. All 48 hybrid states were constructed
and tensor-verified without new training. Replay started at 10:17:55 UTC in
`/dataMeR1/phil/gfm/prodigy-mechanisms-readout`, frozen revision `d804bad0`, tmux
`mechanism-readout-reference`, four CPU threads with GPUs hidden. Both streams
and all five targets/17 decoders completed by 10:39 UTC: 8,160 rows, 480 full-model
cells. There is no new training, random replacement, or target-selected checkpoint.

An independent auditor in `/dataMeR1/phil/gfm/prodigy-mechanisms-verify` at
`212ba556` reread saved weights and checked exact donor tensors, all cached
inputs, and every unchanged upstream probe prediction. It completed by 10:42 UTC.
All 48 saved states match their exact donors; all 320 cached input batches match;
153,600 unchanged upstream prediction-tensor comparisons are bit-exact.
Source/policy/seed effects,
reverse swaps, full-model tradeoffs, and cue comparisons are all declared, not
chosen after a favorable result. The completed analysis code is `7780ddf6`.
At 10:33 UTC, before calculating any swap effects, the secondary cue analysis
was extended to include the already-exported center branch. It retains all
original stages and adds 450 cue cells (1,800 total), plus 576 exact-initial-to-
terminal cue differences. This amendment follows the completed baseline branch
audit; it is not represented as preceding those baseline outcomes. The 8,160-row
swap grid and original directional prediction are unchanged. All 1,800 cue cells
have 128 valid episodes, and the previous 432 terminal cue cells reproduce.

**The predicted universal TwiBot readout rescue fails.** Under the original
lowest-ID/sorted policy, restoring the exact initial readout improves Ukraine's
TwiBot readout probe in all three seeds on both streams, but worsens Hong Kong's
in all three. Means below are three-seed AUC changes, original/fresh streams:

| Target/source | Restore initial readout: probe | Restore initial readout: full model | Implant trained readout in initial background: probe | Implant: full model |
|---|---:|---:|---:|---:|
| TwiBot / Hong Kong | −.04163/−.04579 | −.03563/−.04373 | −.00355/−.00119 | −.11374/−.10940 |
| TwiBot / Ukraine | +.01585/+.00391 | −.03256/−.01578 | −.00748/−.00568 | +.04353/+.05258 |
| Facebook / Hong Kong | +.08403/+.08003 | −.03017/−.02270 | −.07427/−.06940 | +.00482/+.00115 |
| Facebook / Ukraine | +.05616/+.06617 | −.08276/−.08391 | −.04625/−.06826 | +.00278/+.00057 |

These are interventions in a fixed rest-of-network background, not an additive
decomposition of training. Restoring the readout leaves the previously observed
center-branch degradation exactly unchanged. The TwiBot result is consistent
with Hong Kong's trained readout partly compensating for its altered upstream
features, whereas Ukraine's readout imposes a small additional diagnostic penalty.
It does not identify a universal harmful component or establish causal mediation.

Keeping **all four policies and all three seeds** does not erase this distinction.
TwiBot restoration improves Ukraine's probe in 12/12 original and 10/12 fresh
cases (mean +.01379/+.00658), but worsens Hong Kong's in 11/12 and 12/12
(mean −.02389/−.03686). Ukraine's full model instead worsens in 12/12 original
and 10/12 fresh cases (mean −.05055/−.03056). Hong Kong's full-model signs are
mixed, and the reverse full-model swap has large seed/policy variation. Three
policies sharing a training seed are not three additional independent seeds.

**Facebook provides a replicated parameter-level readout penalty for this fixed
probe, but not a model rescue.** Restoration improves the readout probe in all
24 models on both streams; reverse implantation worsens it in all 24 on both.
Across all policies, restoration gains average +.10459/+.10695 for Hong Kong and
+.05508/+.06568 for Ukraine. Yet all 24 full models worsen on the original stream;
on fresh episodes all 12 Ukraine models and 7/12 Hong Kong models worsen. This
separates probe accessibility from the learned downstream decision mechanism.
No conclusion about information-theoretic loss follows from one fixed probe.

The other targets are retained, not hidden nulls. Under the standard policy,
restoring the initial readout reduces COVID Political probe/full AUC for both
sources (full means: Hong Kong −.11700/−.12030, Ukraine −.17655/−.17404), and
Election full AUC by −.30581/−.31083 and −.22603/−.21562 respectively. Suspension
effects remain small and mixed near the weak-input baseline. This is a
target-specific tradeoff, not a generally superior initialized readout.

The added branch audit finds reduced incoming-degree cue agreement already in
the post-convolution center branch for **all 24 models on both streams**.
At the readout, restoring Ukraine's initial weights increases degree agreement
under the standard policy (+.08951/+.08923); restoring Hong Kong's decreases it
(−.13530/−.13505), matching their opposing probe effects. However, Ukraine's
full-model degree agreement also increases (+.04157/+.06467, all three seeds),
while full-model AUC mostly decreases. Increasing agreement with this useful cue
is not sufficient to improve the actual classifier.

Evidence: `data/readout_intervention_{cells,summary}.csv`,
`data/readout_intervention_validation.json`, `data/readout_parameter_effects.csv`, `data/readout_cue_*.csv`,
`data/readout_baseline_cue_changes.csv`, and exact receipts in `data/readout_audit/`.
`figures/readout_{restoration,implant}_tradeoffs.png` shows both directions on
Facebook/TwiBot with every seed, policy, and stream. Other targets and all 17
decoders remain in the complete tables.

The concurrently developed sampler correction (`ece9cb28` in the separate
training-role-exposure worktree) shuffles unique endpoints **before truncation**.
It changes retention and role assignment jointly, rather than being a role-only
change. Its ordered selection law corresponds to the joint uniform/shuffled
policy here, but RNG coupling and full-run protocols differ; do not treat those
new lattice runs as exact numerical replicas or merge them into historical cells.
That worktree and the user's `rolefix-lattice-0906` jobs were left untouched.

## Initial-readout-frozen training: completed; full-model primary fails

The fixed-weight results motivated a different training-time test, recorded
before launch in `paper/planning/readout_training_constraint_2026-09-06.md` outside
git. Eighteen new CPU models compare freely trained and exact-initial-readout-
frozen conditions for Ukraine/Hong Kong/COVID and seeds 0/1/2. Both conditions
are newly trained, with the original lowest-ID/sorted member policy and matched
initialization and complete input tensors. Every other model component adapts
normally. The primary prediction is a **Facebook full-model AUC improvement**;
a probe-only improvement does not satisfy it. All other targets and both streams
remain required. This is a data-informed exploratory follow-up, not untouched-
domain confirmation or an established remedy.

The two-arm three-update toy run and 18-arm eight-update full-graph smoke passed
all input, initialization, optimizer and frozen-weight gates. The latter was
rechecked at `f20e6495`, including a stronger requirement for actual non-readout
parameter updates rather than normalization-buffer changes alone. Eight combined
constraint/analysis tests and all six shared-graph tests passed on Tucker; 31
analysis tests passed locally. Local shared-memory tests hit a sandbox permission
restriction and were therefore run successfully on Tucker, not silently waived.

The substantive pipeline launched **11:04:09 UTC**, frozen Tucker worktree
`/dataMeR1/phil/gfm/prodigy-mechanisms-freeze` at `f20e6495`, tmux
`mechanism-readout-training`, six concurrent CPU models, eight tensor threads and
two loader workers each, with all GPUs hidden. Each model runs 2,500 updates and
keeps 0/100/300/900/2500 checkpoints. The finite continuation evaluates both
128-episode streams only after all 18 training gates pass. Output:
`log/target_mechanisms/readout_constraint_training_20260906`.
The local complete-grid analyzer was committed before outcomes. All 18 models
and training gates completed; evaluation ran from **14:17:53 to 14:25:59 UTC**.
All 3,060 rows, 180 full-model cells, and 1,530 paired changes pass the unchanged
analysis. Every one of the 2,500 full training-input hashes matches within each
free/frozen pair, as do initial weights and consumed member/walk records. Frozen
readout tensors remain exactly initial at every update and saved checkpoint;
other parameters genuinely train. Target inputs match the established cache.

The primary rule required every seed's source-average Facebook full-model gain
and every source's three-seed mean to be positive on **both** streams. It fails:

| Seed | COVID original / fresh | Hong Kong original / fresh | Ukraine original / fresh | Source average original / fresh |
|---|---:|---:|---:|---:|
| 0 | +.01005 / +.00606 | −.00944 / −.02208 | +.00102 / −.00453 | +.00054 / −.00685 |
| 1 | +.01397 / +.01977 | +.01777 / −.01840 | +.01137 / −.00544 | +.01437 / −.00136 |
| 2 | +.02116 / +.01767 | +.03050 / +.01962 | +.00934 / −.01409 | +.02033 / +.00773 |

Values are frozen minus freely trained AUC, not percentage points. The original
stream has three positive seed averages, but fresh episodes have two negative
averages. Hong Kong and Ukraine source means turn negative on fresh episodes.
COVID improves in all three pairs on both streams; this favorable secondary
source result does not replace the unsuccessful all-source primary.

The Facebook readout probe nevertheless improves in **all nine pairs on both
streams**. Mean gains are COVID +.04194/+.06483, Hong Kong +.09316/+.09039, and
Ukraine +.03845/+.03535. Thus the probe/model distinction survives a training-time
intervention in which the downstream network can adapt, not only an abrupt
post-training parameter swap. This rejects the proposed general rescue, not
every possible training constraint or every decoder of these representations.

All other targets expose tradeoffs. Full-model three-seed means are:

| Target | COVID original / fresh | Hong Kong original / fresh | Ukraine original / fresh |
|---|---:|---:|---:|
| COVID Political | +.00892 / +.01112 | +.01668 / +.01384 | −.00107 / +.00399 |
| Election | −.00138 / −.00269 | −.04136 / −.04364 | −.00189 / −.00167 |
| TwiBot | −.00387 / +.00168 | +.00830 / +.01210 | +.01368 / +.00948 |
| Suspension | −.00323 / +.02545 | +.00724 / +.02976 | +.03253 / +.04307 |

On TwiBot, Ukraine's full model improves in all three pairs on both streams;
Hong Kong and COVID have mixed signs. TwiBot readout probes improve for all
Ukraine and COVID pairs but worsen for **all Hong Kong pairs**, on both streams.
Mean probe effects are Ukraine +.03665/+.02465, COVID +.02775/+.02083, and
Hong Kong −.01859/−.02123. The opposing Ukraine/Hong Kong probe directions agree
with the earlier swap diagnostic, but do not imply identical full-model effects.
Hong Kong's Election full-model effects are negative in all three pairs on both
streams. Suspension changes do not establish a strong or prospective predictor.

### Same-seed controls are not exact cross-launch reproductions

All six new freely trained Ukraine/Hong Kong controls differ from their earlier
same-seed standard-policy models. On identical target tensors, the largest
absolute **full-model** AUC change is .09347 (Hong Kong seed 2, COVID Political,
original episodes). Do not conflate the even larger post-metagraph probe changes
with full-classifier changes, or count these controls as additional seeds.

A read-only audit of all six pairs at updates 0/100/2500 establishes:

- At initialization, every model tensor, optimizer state, recorded Python/NumPy/
  Torch RNG state, complete batch-sampler state and resume parameter contract
  matches exactly.
- Recorded parent RNG and complete sampler states still match at 100 and 2500,
  but 31 model-state keys and the optimizer state already differ at update 100.
  Maximum parameter differences are .170–.308 then and 1.077–1.341 at update 2500.
  The much larger terminal differences of 305–2034 belong to metagraph batch-
  normalization running variances, not individual learned weight values.
- Every field in all 2,500 consumed audit rows matches for each control pair,
  including every member's context-node count, not merely its aggregate mean.
  Final walk RNG also matches. Earlier runs did **not** record complete training
  input hashes or worker RNG states. Matching identities and parent RNG therefore
  does not establish matching sampled context tensors/features across launches.
- The production training/model/sampler implementation is unchanged between
  revisions `75f0853f` and `f20e6495`; recorded resolved configurations differ only
  in run identifiers/paths and the new free-condition tag. The launcher adds a
  full-input hashing hook and constraint observer; these are not an independently
  demonstrated explanation for the drift.

A numerical source of update drift is now demonstrated in the bounded test below.
Its quantitative contribution to the cross-launch AUC changes remains unresolved;
unrecorded historical worker/context differences have not been excluded. The
**new** paired full-input audit establishes input matching, but does not establish
deterministic optimization: the paired contrast can still include numerical
execution variability. These are three seeds on already-inspected targets, not
untouched-domain confirmation. Do not silently replace the failed primaries or
claim that the input-matched pairs isolate a noise-free intervention effect.

Evidence: `data/readout_training_{cells,changes,primary,summary}.csv`,
`data/readout_training_validation.json`, `data/readout_training_control_reproduction.csv`,
all ten raw replay files, and complete receipts in `data/readout_training_verified/`.
`control_state_audit.json` retains all 18 saved-state comparisons; the read-only
helper is `setup/target_performance_mechanisms/audit_readout_control_states.py`.
`figures/readout_training_tradeoffs.png` shows all 90 target/source/seed/stream
probe/full-model contrasts, including unfavorable targets and failed primary.

The initial 11:01:24 launch failed before producing a training directory. A
read-only reproduction showed that prepending conda's base bin inside an already
activated child selected `/home/mhchu/miniconda3/bin/python`, which lacks PyG.
The corrected launch explicitly invokes the `prodigy` environment's Python;
the original failure log is preserved with `_failed_environment.log`. The
experiment code, smoke validation, and declared training protocol are unchanged.

### Decoder-localized CPU update nondeterminism — controlled diagnostic

On Tucker's actual PyTorch 2.0.1+cu118 environment with eight CPU threads, a
fixed repeated-index gradient primitive varies across identical executions.
The production S/U/M model also drifts on four fixed synthetic NM batches.
The stronger localization test uses **768-dimensional inputs and the exact
Ukraine seed-0 initial model and AdamW state**, with the production 30-way,
3-support, 4-query, batch-four architecture/loss. Twelve replays cross three
numerical modes with plain/free-audit-hook execution and two repetitions each.
All inputs and initial model digests match; no target-test data or labels are used.

| Numerical mode | First logits | First gradients / updated weights | All four updates across plain/hook repeats |
|---|---|---|---|
| Default production indexing | Bit-exact | Differ; max 2.16e-7 / .003437 | Not exact |
| Deterministic algorithms | Bit-exact | Bit-exact | All logits, gradients and model-state tensors exact |
| Default mode; only two decoder reads use `index_select` | Bit-exact | Bit-exact | All logits, gradients and model-state tensors exact |

Every decoder replacement forward is also checked against the original decoder
on the same tensors: **16/16 bit-exact**. Default plain-versus-plain logit drift
is .06163 by update four (.06744 is the largest across all three default
comparisons). The first weight drift occurs despite gradient differences of
only about 2e-7; it is an observed optimizer amplification, not a prediction
of its eventual target-AUC effect. The free audit hook is not necessary for drift.

This has a concrete library-level explanation. Advanced-index autograd dispatches
to indexed accumulation; the pinned CPU float implementation uses parallel
atomic additions when deterministic algorithms are disabled and serial
accumulation when enabled. Addition order can change floating-point results.
[PyTorch 2.0.1 autograd](https://github.com/pytorch/pytorch/blob/v2.0.1/torch/csrc/autograd/FunctionsManual.cpp#L4514-L4521),
[CPU indexing kernel](https://github.com/pytorch/pytorch/blob/v2.0.1/aten/src/ATen/native/cpu/IndexKernel.cpp#L140-L166).
The decoder-only intervention localizes a sufficient source in this workload;
it does not prove that every other operation is deterministic on every graph,
device, thread count, or software version.

The prespecified **real-source confirmation also succeeds** at runtime `bf18f2d2`.
The first four Ukraine NM training batches hash-identically to the completed
free-control input audit, after restoring its actual step-zero model, optimizer,
sampler and RNG and keeping the original 2500-batch/two-worker contract. Every
sampled real node belongs to Ukraine. Across twelve replays, default first
predictions match but gradients differ by up to 3.30e-7 and first updated
parameters by .003478. Plain-versus-plain logits differ by .31551 at update four.
The largest model-state difference (.26061) is a metagraph BatchNorm running
variance, **not a learned weight**. Both stabilizations again give bit-exact
logits, gradients and complete model state at every update, including hook/plain
comparisons. Their two reference trajectories also have identical digests.
All 16 decoder replacement forward checks pass on these real batches.

The bounded loader emits worker-abort messages during early iterator teardown,
after collecting the four batches. The complete full-input hash checks pass
before replay, all twelve replay results complete, and the sampler/loader
processes have exited. The original log is retained; this is not a failed input
fetch or a silently omitted replay. No further loader inputs are used.

No global production defaults, completed weights, user jobs or historical scores
were modified. This is a reproducibility repair prerequisite, **not a new transfer
remedy or stand-alone ICLR contribution**. The quantitative .09347 historical
AUC difference is still not explained by a four-update test. These real inputs
match the newer free control, whose historical predecessor lacks full input
hashes; do not retroactively claim that predecessor's tensors were verified.

Evidence: `data/control_decoder_numerics.json` (runtime `555a230a`), the earlier
reduced-input check `data/control_update_numerics.json` (runtime `e1ec61e6`), and
`data/control_real_input_numerics.json` (runtime `bf18f2d2`). The validated
`data/control_numerics_{steps,summary}.csv` / validation JSON covers both full-
dimension workloads: 24 replays, 216 tensor comparisons and 32 decoder parity checks.
The latter aggregates comparisons, not independent training seeds.

## Complete mixture complementarity diagnostic — verified

The historical 54-model singleton/pair/LOO lattice has aggregate metrics but did
not retain the complete query predictions needed to distinguish complementary
specialist errors from different decisions learned by a mixture. The new replay
covers all 45 mixtures, all five targets and both fixed episode streams. The
nine specialist exports are reused only after exact input/checkpoint identity
and metric reproduction checks. Equal-probability constituent averaging is
primary; equal-logit averaging is secondary. Neither fits query labels or selects
ensemble weights. Foreign-source pairs and LOO are reported separately; an
ensemble's 2x/8x compute advantage is not evidence of causal training interference.
These are the historical sampler models, not the later corrected runs.

The first replay stopped on one numerical parity failure. A direct matched audit
at `4c89f1b7` established that the Hong Kong-omitted LOO model's suspension AUC
differs by exactly one CPU-tied positive/negative query pair: original GPU
0.5087890625 versus CPU 0.508819580078125. All three GPU repeats reproduce the
original metrics; a fresh unhooked CPU pass reproduces every saved CPU logit
bit-exactly. All predicted classes are unchanged. The original results, failed
replay, pair scores and all repeats are retained. See
`data/mixture_numerical_audit/README.md` and `audit.json`.

The retry launched **11:34:09 UTC**, frozen revision `5fdfba2a`, Tucker worktree
`/dataMeR1/phil/gfm/prodigy-mechanisms-complement`, tmux
`mechanism-mixture-complementarity`. Fourteen tests and the dry run passed.
Its four-thread CPU pipeline uses a generated reference copy for only the audited
cell, preserving the immutable historical table and all general parity tolerances.
The full 7,650-row replay and automatic prediction analysis are complete and
verified: 224 strict original mixture cells plus one individually audited
numerical cell, 540 model cells, 450 fixed-ensemble comparisons and 1,350 error
strata. Local validation passed complete source coverage, all metric arithmetic,
shared-error accounting and both episode streams before interpretation.
Within-target partial rank associations are descriptive; 28 foreign pairs share
sources and do not provide 28 independent training replications.

**Facebook is the clearest lost-complementarity case.** For foreign-source pairs,
the trained mixture trails fixed probability averaging by **2.49/1.88 AUC points**
(original/fresh); 20 of 28 pairs are worse on both streams. The foreign LOO model
trails its eight-specialist ensemble by **6.24/8.17 points**: mixture AUC
.7651/.7368 versus ensemble .8275/.8186. The ensemble also exceeds the best
constituent (.7738/.7680), so this is not solely selection of the strongest donor.
This comparison trains every specialist for 2500 updates. The later budget
check below **does not reproduce the Facebook LOO advantage within the joint
model's total update budget**; do not cite the large gap as a budget-controlled result.
Pair mixtures fix 4.89/5.14% of queries that both constituents miss, but introduce
errors on 7.22/7.01% that both get right. This accounts for nearly all their mean
accuracy disadvantage; it is an accuracy decomposition, not an AUC decomposition.

**COVID Political differs.** Foreign pair mixtures outperform probability
averaging by **1.97/1.44 AUC points** on average; 16 of 28 win on both streams.
Their positive mean accuracy difference comes from disagreement cases, despite
a negative net contribution on unanimous-correct/shared-wrong queries. Joint
training is not uniformly worse than combining specialist predictions.

TwiBot's mean mixture-versus-ensemble AUC differences are small and negative
(pairs −.10/−.18 points, LOO −.07/−.36). Election changes sign across streams;
suspension remains near chance. Disagreement alone is not a stable source-quality
explanation: COVID Political's strong raw association with gain over the mean
constituent (.760/.727) shrinks to .021/−.016 after descriptive adjustment for
constituent mean and AUC range. No independent-pair significance is claimed.

These outcomes support **target-dependent information combination**, not a causal
proof of training interference or a general remedy. The ensemble uses 2x/8x total
training updates and model forwards; checkpoint diversity, calibration, capacity
and reduced per-source exposure in fixed-budget joint training remain unresolved.
Both streams reuse the same five inspected domains and one training seed.
Evidence: `data/mixture_complementarity_{summary,comparisons,associations,stream_agreement}.csv`,
`data/mixture_complementarity_predictions/`, and
`figures/mixture_complementarity_foreign.png`.

## Corrected production sampler: relevance check complete

A read-only audit found that the separate corrected lattice pipeline completed
all nine singleton trainings at 10:28:43 UTC, then stopped before NM evaluation
or pair/LOO training because its reference ledger pointed into a removed worktree.
The original log is preserved in `data/corrected_sampler_verified/source_pipeline.txt`.
The ledger exists in the runtime archive, but blindly replacing its path is not
a verified fix: the corrected member selector also changes generated NM test
episodes. Our work leaves that pipeline untouched and reuses its checkpoints
for the established classification protocol instead.

All 36 checkpoint/training-state pairs (nine sources × 100/300/900/2500 updates)
passed finite-state, common-architecture, optimizer-step, source-restriction and
effective-config checks. A separate CPU replay launched **11:43:00 UTC**, frozen
revision `311383ab`, worktree `/dataMeR1/phil/gfm/prodigy-mechanisms-corrected`,
tmux `mechanism-corrected-sampler`, four threads with GPUs hidden. Five targets,
17 decoders and both fixed episode streams are complete. All 6,120 cells, 320
cached-batch comparisons, raw-probe equivalence, weight identities, source/step
coverage and original/fresh input checks passed before interpretation.

The question is whether the earlier target-specific bottlenecks and checkpoint
trends survive the newer production recipe. This is **not** a role-only causal
contrast: shuffling before truncation changes retained identities and roles
jointly, with different random-number consumption. Training inputs and initial
weights are not guaranteed matched to the historical runs. The three-seed
factorial remains the controlled experiment; this one-seed reuse addresses
relevance to the corrected models, not fresh-domain or independent-seed proof.

**The TwiBot pooled-probe/full-model mismatch survives.** From update 100 to
2500, all nine corrected models improve pooled ridge AUC on both streams
(mean +.05129/+.05577), while all nine full models decline
(−.08177/−.09296). Restricting to the eight foreign sources gives
+.04735/+.05423 for pooling and −.08961/−.09887 for the full model. The learned
readout probe declines for 7/9 original and 8/9 fresh models (mean
−.01646/−.02589). These are fixed-checkpoint trends, not a claim that training
from initialization is universally harmful; no exact historical step-zero
reference exists for this comparison.

**A stronger earlier center-branch statement does not transfer unchanged.**
The historical center probe declined for every source, but corrected center
changes are positive for 6/9 on original episodes and only 1/9 on fresh episodes
(means +.00462/−.00842). Its endpoint change is less negative than historical
for all nine sources on both streams. The corrected center/readout/full trends
must be kept distinct; pooled and center branches remain parallel computations.

**Facebook's raw-input gap also survives.** Raw-description ridge AUC is
.91961/.93011, versus a best corrected full model of .77827/.76330 and a
nine-source mean of .70348/.69247. All foreign-source full models improve from
100 to 2500, but their terminal mean is slightly below the historical recipe
(−.00937/−.00906). Correcting sampling does not close the gap between the raw
description probe and the trained classifier.

At the terminal checkpoint, foreign-source mean full-model changes from the
historical recipe are small and target-dependent: COVID Political
−.00295/−.00438; Election +.00315/−.00066; TwiBot +.00825/+.00358;
suspension +.00413/+.00291. Foreign donor rank correlations are high for
COVID Political (.929/.976) and TwiBot (.952/.881), weaker or stream-sensitive
elsewhere. None of these unmatched single-seed differences establishes a causal
benefit of the sampler correction or explains general donor quality.

Evidence: `data/corrected_sampler_{cells,vs_historical,endpoint_changes,endpoint_summary,terminal_ranking}.csv`,
`data/corrected_sampler_verified/`, and
`figures/corrected_sampler_twibot_trajectories.png`.

## Specialist training-update budget check — complete, primary prediction failed

The exploratory protocol was recorded after terminal ensemble/trajectory results
were known but before computing earlier-checkpoint ensembles. All four saved
specialist steps (100/300/900/2500), all 45 fixed 2500-update mixtures, five
targets and both streams were retained. All 54 actual training configurations
were resolved against completed run logs/results and exact checkpoint paths:
source restrictions, four episodes per batch, 2500 total updates and the explicit
four-step save schedule agree. This is not a checkpoint-name-only budget claim.

The CPU prediction-only job ran in `/dataMeR1/phil/gfm/prodigy-mechanisms-budget`,
frozen revision `1fee2c57`, tmux `mechanism-mixture-budget`, launched
**12:12:14 UTC** and verified complete by 12:14:56 UTC. Ten tests and the audited
dry run passed on Tucker. All 810 distinct prediction cells reproduce saved
metrics and exact cached inputs; 1800 comparisons and 5400 error strata pass
local arithmetic/coverage checks. Every step-2500 model/ensemble/stratum result
reproduces the earlier complementarity analysis, with no relaxed tolerances.

The fixed rule chooses the largest available specialist step with K × step ≤
2500: **900 per pair member (1800 total)** and **300 per LOO member (2400 total)**.
The primary prediction was a positive foreign Facebook LOO ensemble advantage
on both streams. It **fails on both**: ensemble AUC .73746/.73412 versus joint
model .76510/.73685, differences **−2.764/−.272 AUC points**. At 900 updates
per specialist (7200 total), the ensemble rises to .80975/.79615; at 2500 each
(20000 total), .82754/.81856. The large terminal Facebook advantage cannot be
used to rule out extra training budget. Its source/capacity/optimization causes
remain entangled; this observation is not proof that extra updates cause the gap.

Secondary outcomes under the same fixed rule, ensemble minus trained-mixture
AUC points (original/fresh):

| Foreign target | Pair mean, 28 cases | LOO, one case |
|---|---:|---:|
| COVID Political | −1.861 / −1.384 | −2.993 / −1.827 |
| Election | −.277 / −.057 | +.311 / −.433 |
| Facebook pages | +.975 / +.251 | −2.764 / −.272 |
| TwiBot | +.806 / +1.022 | +7.526 / +7.791 |
| Ukraine suspension | +.388 / +.039 | +1.422 / +1.324 |

**TwiBot supplies a secondary within-update-budget advantage, not a replacement
primary success.** At step 300, its foreign ensemble scores .70012/.68524
versus the LOO model's .62486/.60733. It also beats the best same-step
constituent by .02243/.00971. Continuing the specialists to steps 900 and 2500
reduces ensemble AUC on both streams; their gain falls to .07/.36 AUC points
at step 2500. Foreign Facebook pair ensembles have a much smaller positive
mean at step 900, with 13/28 positive on both streams; COVID Political still
favors the trained mixtures. Suspension remains close to chance and Election
changes sign across streams.

All budget comparisons keep ensemble capacity and model forwards **2x/8x**.
Saved-update/episode counts are not matched training FLOPs or wall time, since
sampled subgraphs differ. No query-fitted weights, temperatures, or checkpoint
selection were used. One seed, shared sources and already-inspected targets
preclude independent replication or a general remedy claim. The joint model
is fixed at step 2500; these curves do not establish a matched mixture-trajectory
causal contrast or a new explanation of overall Ukraine/COVID donor quality.

Evidence: `data/mixture_budget_predictions/` (including actual-config audit),
`data/mixture_budget_{comparisons,summary}.csv`, `data/mixture_budget_validation.json`,
and `figures/mixture_training_budget_foreign.png`.

## Restoring training-time attention multiplicity does not rescue transfer

The prospective count-restoration diagnostic is complete: **630 full-model
cells**, nine historical singleton checkpoints, five targets, seven conditions,
and both 128-episode streams. No encoder passes or training were needed. Reusing
cached pre-metagraph embeddings with the production metagraph and cosine decoder
reproduces **all 2,880 baseline batches bit-exactly**, both saved post-metagraph
embeddings and full-model logits. Unhooked restoration also reproduces every
baseline; all checkpoint digests, input tensor hashes and baseline metrics pass.

The hypothesis was that moving from 30-way/3-shot training to 2-way/10-shot
evaluation changes the attention balance in a harmful way. Known counts define
weights, without query-label fitting: label destinations give positive supports
weight .3 and negative supports 8.7; query destinations give each incoming label
weight 15; self loops keep weight one. Adding log(weight) before softmax changes
attention only. A separate virtual-multiplicity condition also changes the
aggregate affine bias to sum(weight)*bias, and a bias-only control separates it.
Way-only, shot-only and reciprocal-direction controls are all retained.

**The primary prediction fails in all four target/stream cells.** Joint
attention-only restoration must improve the eight foreign donors' mean full-model
AUC and beat reciprocal-direction restoration for each of Facebook and TwiBot
on both streams. Instead, restoration worsens the mean on every target/stream:

| Target, eight foreign donors | Joint attention ΔAUC points, original/fresh | Reciprocal control ΔAUC points, original/fresh |
|---|---:|---:|
| COVID Political | −.532 / −.582 | +.369 / +.365 |
| Election | −.169 / −.364 | +.114 / +.071 |
| Facebook pages | −.712 / −.631 | +.400 / +.357 |
| TwiBot | −.484 / −.528 | +.426 / +.329 |
| Ukraine suspension | −.126 / −.171 | +.449 / +.373 |

Only 1/8 and 2/8 Facebook donors improve on original/fresh episodes, respectively;
TwiBot has 4/8 and 2/8. The favorable reciprocal control does not replace the
failed primary prediction. Its small average gains do not close the prior probe/
full-model gaps or establish a remedy, and individual donor signs vary. The
way-only condition closely tracks the joint effect, whereas shot-only effects
are small. Including virtual-multiplicity bias does not rescue either primary
target: Facebook −.778/−.686 points and TwiBot −.285/−.221 points.

This is not a null manipulation. Averaged over the nine donors, both streams,
batches and heads, Facebook label-node negative-support attention mass rises
from .1965 to .3281, and query attention to label nodes rises from .1034 to .2176.
TwiBot's corresponding changes are .1952 to .3275 and .1083 to .2302. Every
per-head destination-group mass audit sums to one. Changed attention is not
equivalent to useful task-specific computation.

The production one-layer graph also explains why **other-query count is not a
direct inference dependency**, conditional on fixed supports, label vectors,
query inputs and evaluation normalization: query messages do not enter label
nodes, and each query attends only to initial label nodes and itself. Adding a
query leaves existing outputs unchanged in the algebraic test for all seven
conditions. This does not make episode distribution irrelevant during training.

These controls reweight existing examples; they do **not** create 30 distinct
classes or diverse new negatives, change training normalization, or constitute
a matched-training cardinality ablation. They weaken this precise attention-mass
restoration explanation, not all train/test episode shift. One training seed and
previously inspected targets preclude untouched-domain confirmation. The
Ukraine/COVID donor-quality mechanism remains unresolved.

Evidence: `data/episode_cardinality_replay/`,
`data/episode_cardinality_{cells,summary,attention}.csv`,
`data/episode_cardinality_validation.json`, and
`figures/episode_cardinality_foreign.png`. The complete per-batch/head attention
audit is losslessly compressed; query predictions remain on Tucker.

## Artifacts and runtime provenance

- Local branch/worktree: `codex/target-performance-mechanisms`,
  `/Users/philipp/projects/gfm/prodigy-mechanisms`.
- Episode-cardinality diagnostic: frozen `b882d95a` in Tucker
  `/dataMeR1/phil/gfm/prodigy-mechanisms-count`, output
  `log/target_mechanisms/episode_cardinality_20260906`, tmux
  `mechanism-episode-count`, eight CPU threads, GPUs hidden. Launched
  12:36:34 UTC and verified complete by 12:38:50 UTC on September 6. Nine
  count/algebra/primary-rule tests passed on Tucker before launch; the expanded
  local analysis suite has 49 passing tests. Active training worktrees were not
  changed. The prospective design remains in the sibling paper planning tree.
- Three-target replay: Tucker `prodigy-mechanisms/log/target_mechanisms/specialist_cpu_20260906`;
  code revision `97cc7704`. Partial run, three targets complete.
- Source audit: Tucker `prodigy-mechanisms-audit/log/target_mechanisms/source_sampler_20260906_v2`;
  revision `1308f9b0`, all nine sources complete and `DONE` present.
- Continuation: output `specialist_cpu_tail_20260906`, both targets complete and
  `DONE` verified.
- Architecture control: output `meta_bias_cpu_20260906` in
  `prodigy-mechanisms-audit`; all 45 cells complete and `DONE` verified.
- Continuation and architecture control used revision `b260cd56`, Tucker CPU,
  eight threads each. Input probes used revision `6418c78d`, four CPU threads.
- Matched-pair stage replay: `pair_stage_cpu_20260906_v2`, revision `b260cd56`,
  all 28 cells complete. The first attempted launch used an incompatible device
  token and stopped before evaluation; that partial directory is preserved.
- Fresh replay and scalar probes: `fresh_stage_cpu_20260906` and
  `fresh_input_scalars_20260906`, revision `be98b600`, both complete.
- Scalar follow-up and source coverage in the audit worktree:
  `input_scalars_20260906`, `source_coverage_20260906`, revision `be98b600`,
  both complete. All work above used CPUs.
- Suspension metadata diagnostic: audit worktree `metadata_probe_20260906`,
  revision `0e41eb23`, both episode streams complete. All row/feature/label
  alignment gates passed. The harmless pandas read-only-array warning involved
  no in-place writes; the helper now explicitly copies that array.
- Size-adjusted coherence/incoming-presence follow-up:
  `cue_controls_{original,fresh}_20260906`, revision `8c227089`, both complete.
- Fresh fixed-input perturbations and joint prompt-label erasure: all 60 cells
  complete, main mechanism worktree at `4fc6c7bd`.
- A bounded **20-update timing run only**, with GPUs explicitly hidden, completed
  in the training worktree at `ecd48890`: `cpu_timing_20260906`. It uses the exact
  full source graph, eight tensor threads, and four loader workers. It is not one
  of the 24 substantive models. Setup took 175 seconds; steady training took
  1.626 seconds/update. All 20 consumed steps and finite updated weights passed.
- A 24-arm **concurrent CPU smoke only** passed in the training worktree,
  revision `75f0853f`, `member_cpu_smoke_20260906`: 20 updates per arm, six active
  models, eight tensor threads and two loader workers each (60 total). All 24
  finite-weight, initialization, consumed-anchor, member-set, and walk-RNG gates
  passed. Steady throughput was 1.37–2.06 seconds/update across arms.
  `data/cpu_concurrent_smoke_receipt.json` preserves the validity receipt.
- Substantive 24-arm CPU training started **2026-09-06 05:52 UTC**, same frozen
  training revision and resources, output `member_cpu_training_20260906`, tmux
  `mechanism-member-cpu`. The prospective CPU execution amendment was recorded
  before launch. All 24 runs completed at **09:55:58 UTC**; all consumed-stream
  gates passed at **09:56:39 UTC**. Every declared checkpoint (0/100/300/900/2500;
  120 files) is retained.
  Initial weights match even across the two sources within each seed. An
  independent audit confirms all effective configurations match the declared CPU
  recipe, all non-treatment settings are common, and all consumed episode-source
  labels identify the intended source. No target-policy effects are interpreted
  before both complete evaluation streams pass. All GPUs were hidden.
- Fresh-episode matched-pair stage replay: all 28 baseline cells complete in the
  main mechanism worktree at `4fc6c7bd`, `fresh_pair_stage_cpu_20260906`.
- All-nine-source, four-checkpoint trajectories on both episode streams:
  `trajectory_{original,fresh}_20260906` in the main mechanism worktree, frozen
  revision `3c88bdbfeafacc49464afbacea8250ddf673d9b1`, four CPU threads. Both
  `DONE` markers and all checkpoint/input/terminal-reference gates passed.
- Finite training-to-evaluation continuation completed in audit worktree tmux
  `mechanism-member-eval`, frozen revision `c23bee34`. It requires all 24 CPU
  training validity gates, then evaluates original and fresh episodes and checks
  all cached tensors against established references. All gates passed; the smoke
  experiment is not a research result. Output: `member_evaluation_20260906`.
- Actual consumed exposure (10,000 episodes per model, three seeds): uniform
  retention changes Hong Kong's repeated-member fraction from .39510 to .29795
  and effective member count from 74.63 to 148.62; Ukraine changes from .03536 to
  .02242 and 2,516 to 4,006. The manipulation works but does not equalize the
  source distributions. Consumed context means are 67.42/68.89 for Hong Kong and
  79.06/78.83 for Ukraine (lowest/uniform); these are exposure-weighted actual
  training contexts, not the earlier small uniform-unique-member context sample.
  These manipulation checks alone are not evidence of a target-performance gain.
- Post-hoc TwiBot cue agreement ran locally using existing prediction-only tensor
  exports (~44 MB; no graph-feature export), code `a1c0aa4b`. All 648 cells passed
  exact cached-input checks; two rank/orientation/constant-score unit tests passed.
  This new analysis remains local pending approval to publish to the public remote.
- Frozen label-interface controls completed in a detached private worktree,
  `/dataMeR1/phil/gfm/prodigy-mechanisms-followup`, revision `1010d57c`, tmux
  `mechanism-label-interface`, four CPU threads. All 16 scoped-hook/replay tests
  passed locally and on Tucker. Both streams and all five targets are fixed in
  advance; both `DONE` markers and all input/weight/baseline/norm gates passed.
  Code reached Tucker by **direct private git transport**, without any
  public GitHub update; active training/evaluation worktrees were not changed.
- The label-table/projection audit reads all 36 existing checkpoint tensors;
  `data/label_interface_checkpoint_inventory.json` records exact digests and norms.
  Both components are byte-identical across every historical source and step.
- Cross-task replay of 15 pre-existing compatible eight-source NM interventions
  completed in the main mechanism worktree at frozen revision `3c88bdbf`, tmux
  `mechanism-campaign-cls`, four CPU threads. Its 30 declared checkpoints separate
  a common 6,000-update rule from original source-validation selection. All 30
  strict finite-weight/forward checks passed, and all 15 selected checkpoint-file
  hashes match their original NM results. Three forward-incompatible models are
  explicitly excluded. Both `DONE` markers and every input/weight/original-NM
  checkpoint-file gate passed. Manifest revision `6732081a`; no new training or target-
  selected checkpoints. Only TwiBot is an unseen source. This is a one-training-
  seed exploratory reuse, not a substitute for the completed 24-arm factorial.
- Exact-initialization reference completed in the unchanged private follow-up
  worktree at `1010d57c`, four CPU threads, tmux `mechanism-initial-reference`:
  `initial_reference_{original,fresh}_20260906`. Both `DONE` markers, all starting-
  weight digests, raw-probe matches, and every cached-input gate passed. Only
  the checkpoint manifest was transferred as data; no source files were copied.
- The 24-arm member-selection intervention (two sources × four policies × three
  training seeds) is implemented at `5e0a3537`. All 24 lightweight tests and a
  four-policy, three-update toy CPU integration passed on Tucker. These are
  implementation checks, not substantive training results. A fail-closed consumed-stream verifier is
  implemented at `c4cd97aa`; its 12 combined diagnostic/verifier tests passed
  locally and on Tucker. No user GPU jobs were interrupted.

Compact evidence: `data/replay_cells.csv`, `data/replay_stage_auc.csv`,
`data/replay_intervention_deltas.csv`, `data/replay_validation.json`,
`data/source_sampler_summary.json`, and `data/source_sampler_protocol.json`.
Runtime tensors/identities and large feature exports remain on Tucker.
