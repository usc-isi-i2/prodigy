# First mechanism results: input signal, learned readout, and sampled task quality

2026-09-06 UTC. Exploratory, training seed 0. This is not a causal source-ranking
claim or a completed ICLR result. Three classification targets are complete here;
the remaining two and a targeted metagraph control are running on Tucker.

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

For these 27 pretrained baseline cells, reference episode hashes match and the
largest aggregate metric difference from historical GPU evaluation is 2.83e-7.
Trace/no-trace logits are bit-identical on the same device. The cached-batch hash
also covers features and all metagraph tensors. Historical fingerprints already
included sampled context IDs and edges, not just centers/labels.

The first TwiBot cell stopped the larger sweep: CPU/GPU AUC differed by 1.70e-6,
while accuracy/F1 and episode hash matched. Its two-target continuation explicitly
allows at most 1e-5 AUC portability error, keeps decision checks at 1e-6, and
records observed errors. The failed partial run is preserved; it is not reported
as a completed five-target sweep.

## Target inputs versus model use

Same-query AUC, with the same support-only ridge decoder for the first two columns:

| Target | Raw center features | Raw neighborhood mean | Best full specialist |
|---|---:|---:|---:|
| COVID Political | .8231 | .8629 | .9464 (Ukraine) |
| Facebook Page Reference | .9196 | .7531 | .7738 (TwiBot) |
| Election2020 | .8401 | .9868 | .9909 (TwiBot) |

This directly distinguishes readily decodable input signal from final PRODIGY
performance. Facebook's poor result cannot simply be blamed on absent signal in
the supplied inputs. Conversely, Election's near-ceiling results largely reflect
label information available in sampled neighborhood features before NM training.
These are 10-shot episodic probes, not fully supervised references or Bayes limits.

### The Ukraine–TwiBot contrast is stage-dependent

| Target / stage | Ukraine | TwiBot | Ukraine minus TwiBot |
|---|---:|---:|---:|
| COVID Political: S pooled features + ridge | .9268 | .9269 | −.0001 |
| COVID Political: learned readout + ridge | .9444 | .9373 | +.0071 |
| COVID Political: full model | .9464 | .9185 | +.0279 |
| Facebook: S pooled features + ridge | .8484 | .8704 | −.0220 |
| Facebook: learned readout + ridge | .8048 | .8232 | −.0184 |
| Facebook: full model | .7720 | .7738 | −.0018 |

The singleton Facebook difference is tiny; the preceding lattice audit found a
larger consistent reversal among six matched pair compositions. Those pair
checkpoints still need this stage-level replay. Do not substitute the singleton
contrast for the pair evidence.

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

Zeroing raw label-text embeddings barely changes Ukraine/TwiBot on these targets.
Erasing support-label relations instead brings them near chance. Target-batch BN
moments do not close the Facebook gap. Those BN tests use query covariates and
are explicitly transductive, not valid leakage-free adaptation claims.

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

## Targeted architecture hypothesis under test

The metagraph projects each attention-weighted message with an affine map, then
sums. Its output-projection bias is therefore added once per incoming edge, not
once per destination. An analytic unit test proves this term. Default example
degrees change from 31 in 30-way NM to 3 in binary CLS; label-node degrees change
from 91 to 21 with the current support counts.

A diagnostic now removes only that excess bias, holding all attention coefficients
and learned weights fixed. It may improve, harm, or leave transfer unchanged;
the implementation detail alone is not evidence that it explains performance.
If promising, it needs matched retraining and independent episode/training seeds.

## Artifacts and current jobs

- Local branch/worktree: `codex/target-performance-mechanisms`,
  `/Users/philipp/projects/gfm/prodigy-mechanisms`.
- Three-target replay: Tucker `prodigy-mechanisms/log/target_mechanisms/specialist_cpu_20260906`;
  code revision `97cc7704`. Partial run, three targets complete.
- Source audit: Tucker `prodigy-mechanisms-audit/log/target_mechanisms/source_sampler_20260906_v2`;
  revision `1308f9b0`, all nine sources complete and `DONE` present.
- Continuation: tmux `mechanism-replay-tail`, output `specialist_cpu_tail_20260906`.
- Architecture control: tmux `mechanism-meta-bias`, output `meta_bias_cpu_20260906`
  in `prodigy-mechanisms-audit`.
- Both new jobs use revision `b260cd56`, Tucker CPU, eight threads each. No user
  GPU jobs were interrupted. Both worktrees must stay at their revisions while running.

Compact evidence: `data/replay_cells.csv`, `data/replay_stage_auc.csv`,
`data/replay_intervention_deltas.csv`, `data/replay_validation.json`,
`data/source_sampler_summary.json`, and `data/source_sampler_protocol.json`.
Runtime tensors/identities and large feature exports remain on Tucker.
