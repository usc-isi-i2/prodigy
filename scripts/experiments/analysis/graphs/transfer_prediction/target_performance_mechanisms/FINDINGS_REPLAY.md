# First mechanism results: input signal, learned readout, and sampled task quality

2026-09-06 UTC. Exploratory, training seed 0. This is not a causal source-ranking
claim or a completed ICLR result. All five classification targets, a fresh-episode
replication, matched-pair stage replay, input probes, source coverage, and the
targeted metagraph control are complete. Controlled retraining is running; its
outcomes are not available yet.

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

Neither result alone establishes why changing a training source causes the final
target performance. The next controlled retraining should test member selection
and label conflicts, while fixed-query replay tests representation/decoder use.

## Exact comparison contract

Nine specialists plus one deterministic random initialization, identical cached
128-episode 2-way/10-shot test inputs within each target. Probes use only each
episode's support labels. Ridge regularization is fixed at 1 after row-wise
feature normalization; no query labels select it. No model weights are updated.

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

TwiBot's best full donor changes from Ukraine to Midterm on the fresh stream;
the Ukraine/TwiBot singleton Facebook ordering also changes. Do not present those
rankings as stable conclusions. These are two episode streams from the same
domains and frozen training seed, not independent training or domain replications.

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

Zeroing raw label-text embeddings barely changes Ukraine/TwiBot on these targets.
Erasing support-label relations gives much worse, roughly chance-level results
on COVID Political and Facebook, but **not universally chance**: Election retains
strong positive or inverted rankings with label-text embeddings still present.
A joint erasure of support relations and label text gives exactly .5000 AUC in
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

## A targeted architecture hypothesis that did not explain the gap

The metagraph projects each attention-weighted message with an affine map, then
sums. Its output-projection bias is therefore added once per incoming edge, not
once per destination. An analytic unit test proves this term. Default example
degrees change from 31 in 30-way NM to 3 in binary CLS; label-node degrees change
from 91 to 21 with the current support counts.

A diagnostic removed only that excess bias, holding all attention coefficients
and learned weights fixed. It completed all 45 specialist-by-target cells with
matching reference episode fingerprints. The results are nearly unchanged;
the exact paired deltas are in `data/meta_bias_deltas.csv`. For example, Ukraine
on COVID Political changes from .9464 to .9465 and TwiBot from .9185 to .9179.
This control does not explain the large observed transfer gaps. Do not prioritize
a bias-placement training intervention on the strength of the implementation
detail alone. The negative result is useful: it removes one plausible distraction.

## Artifacts and runtime provenance

- Local branch/worktree: `codex/target-performance-mechanisms`,
  `/Users/philipp/projects/gfm/prodigy-mechanisms`.
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
  before launch. No policy-effect results exist yet. Do not update this worktree
  while it runs. All GPUs are hidden from these processes.
- Fresh-episode matched-pair stage replay: all 28 baseline cells complete in the
  main mechanism worktree at `4fc6c7bd`, `fresh_pair_stage_cpu_20260906`.
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
