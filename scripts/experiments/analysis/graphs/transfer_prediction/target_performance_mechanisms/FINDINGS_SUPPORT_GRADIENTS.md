# Matched support-label binding: a learned, source-dependent effect

## Result

Making repeated support accounts carry consistent soft labels changes the
learning signal, but it is **not an established repair**. At the trained
checkpoints, it raises NM loss for all three sources on average. Relative to an
equally sized, class-matched control, it is less harmful for Hong Kong and COVID
but more harmful for Ukraine. Every seed's four-batch mean has that source-specific
direction, under both production training normalization and frozen metagraph
normalization. Thus Ukraine and COVID do not share a simple response to this
support-conflict manipulation, despite both being strong political donors.

The experiment also verifies a training-only path: changing support label-edge
values changes query representations through metagraph BatchNorm. Freezing only
that normalization eliminates the query-representation change exactly, while
the source-dependent loss contrast remains. This path is not itself a sufficient
explanation of the contrast, nor a demonstrated cause of target performance.

No optimizer updates or target evaluations were performed. This is a controlled
fixed-weight gradient diagnostic, not a new trained method or a transfer result.

## Inputs and comparison

Nine completed free-readout controls: Ukraine/Hong Kong/COVID × seeds 0, 1, 2.
For each, reconstruct the **first four actual consumed batches** from its saved
step-zero sampler/RNG state. All 36 full tensor hashes match the original
training input audit, including context features and edges. Each batch has four
30-way episodes, three supports and four queries per class: 360 supports and
480 query occurrences. These prefixes are not a representative sample across
the whole training run. They contain 144 episode occurrences across nine models.

Measure gradients at saved checkpoints 0 and 2500, without taking a step.
Except for the first batch at initialization, these are counterfactual gradient
probes at fixed states, not the gradients observed during historical training.
The actual TrainerFS forward, full gradient inventory and resulting running
buffers match the lightweight reconstruction bit exactly for each model's first
initial batch. The training configuration bypasses the text-label projection
and uses a frozen learned-label table; the projection's gradient block is
reported as inactive, not as an observed zero gradient.

Three conditions:

1. **Baseline:** original hard support relations.
2. **Identity-soft:** when one center appears as a support under k pseudo-classes,
   replace those k occurrences' signed label-edge vectors by their mean. Each
   copy now supplies the same soft relation to the same set of classes.
3. **Permuted-soft control:** move those soft vectors by a seeded nonzero cyclic
   shift among the three original supports of each class. This preserves every
   class's multiset of support-label vectors and the perturbation magnitude, but
   changes which identities/context representations receive them.

Both changes preserve all graph inputs, query truth, roles, edge counts and,
for each label node, its total signed support-edge weight over the episode.
Query-support and query-query identity ambiguity
are not removed. A query-truth tamper leaves both intervention plans unchanged.
The control can accidentally retain some fully identity-consistent groups:
the mean affected-position fraction is 1.59% for Hong Kong, 1.14% for Ukraine,
and 0% for COVID. It is a close matched control, not a guarantee of zero repeated
identities. Soft edges are a counterfactual change in supplied support evidence;
they are not corrected semantic annotations.

Cross all conditions with production training-mode normalization and a second
mode freezing **only metagraph BatchNorm** to the checkpoint's saved buffers.
Encoder BatchNorm remains in training mode in both. Restore the exact state
before every forward, and use deterministic CPU operations.

## Trained-state NM loss changes

At checkpoint 2500, means across three seeds and four actual prefix batches;
positive changes mean worse loss. These are training-task NLL values in nats,
not target-test AUC changes:

| Source | Identity-soft minus baseline | Matched control minus baseline | Identity-soft minus control |
|---|---:|---:|---:|
| COVID | +.01466 | +.03263 | −.01797 |
| Ukraine | +.05289 | +.01791 | +.03499 |
| Hong Kong | +.32720 | +.59347 | −.26628 |

The identity-versus-control contrasts by seed are:

| Source | Seed 0 | Seed 1 | Seed 2 | Number of individual batches with negative contrast |
|---|---:|---:|---:|---:|
| COVID | −.01021 | −.01741 | −.02628 | 11/12 |
| Ukraine | +.03320 | +.04908 | +.02268 | 1/12 |
| Hong Kong | −.25306 | −.33448 | −.21129 | 12/12 |

At initialization, the maximum absolute identity-versus-control NLL difference
over all inputs and both normalization modes is only 4.53e−6. The large
trained-state differences therefore emerge with the learned state; they are
not present at comparable magnitude in the shared-per-seed initialization.
This does not isolate which updates or source property caused their emergence.

The same trained-state source signs survive frozen metagraph normalization:
COVID −.01791, Ukraine +.03422, Hong Kong −.24908; every seed mean again has
the corresponding sign. In particular, the Hong Kong contrast cannot be
attributed entirely to the training-only query-normalization path.

The active full-parameter gradients differ between the two matched label
bindings. Under production normalization, mean relative gradient distances are
.406 for COVID, .544 for Ukraine and .516 for Hong Kong, where the denominator
is the **identity-soft** gradient norm. Mean directional cosines are .920,
.840 and .915 respectively. These are differences between counterfactual
learning signals, not measured downstream benefit or gradient opposition to a
target task. Active-block and every-batch results are retained separately.

## Normalization carries a support-label change into training queries

All altered conditions leave pre-metagraph embeddings exactly unchanged.
At trained checkpoints, support-only label changes alter query post-metagraph
vectors in all 36 input/model cells for identity-soft and all 36 for the control.
The identity-soft maximum absolute coordinate change per batch ranges from
.0355 to 1.4373. These are representation coordinates, not probabilities.

With only metagraph BatchNorm frozen, those query vectors are **bit exact** in
every corresponding condition. The updated label vectors may still differ.
This isolates the additional query-representation path during training, distinct
from the label-only route in the earlier fixed-evaluation role study. It does
not imply query ground-truth leakage: no query truth enters the intervention.
The persistent loss contrast after freezing also prevents calling normalization
the sole mechanism behind identity-specific sensitivity.

## Gradient cancellation is not the missing simple explanation

For repeated support identities we measured norm(sum of embedding gradients)
divided by sum of their norms, and compared with the same-sized, class-matched
control groups. At the trained baseline under production normalization, the
mean identity-minus-control ratio is −.00629 for Hong Kong and −.00611 for
Ukraine, with a positive seed-1 mean in both. COVID's difference is +.01344.
This does not show a stable, disproportionately strong Hong Kong cancellation
effect beyond the matched grouping control.

These are gradients with respect to **separate support embeddings**. Their sum
describes what would happen if those latent representations were tied. Different
sampled contexts have different encoder Jacobians, so this is not automatically
cancellation of actual encoder-parameter gradients. The full-parameter contrasts
above are a different, directly measured quantity.

## Validity, limitations and next decision

All 432 cells are complete; 144 repeated baseline forward/backward probes are
bit exact. An independent pass over saved raw tensors recomputes every digest,
loss/cohort statistic, parameter-gradient contrast and group statistic. All 36
input hashes and intervention matching checks are reverified. The independent
local analysis checks the complete grid, matched groups, cohort conservation and
inactive-gradient handling. Eight runtime tests pass on Tucker; four new
completed-evidence analysis tests pass locally. The local spawned-worker test
cannot run inside the laptop shared-memory sandbox; it passes on Tucker.

The initial export failed during early worker shutdown after writing its checked
prefixes. A finite-dispatch loader fixed termination without relaxing input hashes.
The first gradient smoke failed while summarizing the inactive projection block.
Both failed attempts are retained and excluded. The revised smoke and full
campaign pass; completed model checkpoints were never modified.

Input export runtime: `727722e8`; successful smoke/full gradient runtime:
`83befe22`. Tucker artifacts:
`/dataMeR1/phil/gfm/prodigy-mechanisms-role/log/support_identity_inputs_v2_20260906`
and `.../support_identity_full_20260906`. The compact validated exports live in
`data/support_identity_gradients/`; prefix/trainer receipts are retained in
`data/support_identity_inputs/`. Rebuild the local tables with
`analyze_support_identity_gradients`; full tensors stay on Tucker.

**Decision:** do not launch a large label-softening training sweep as an assumed
repair on this evidence. The test establishes a learned, source-specific
support-label-binding effect but weakens a common Ukraine/COVID donor explanation
based on duplicate-support ambiguity alone. A next bounded check should establish
what repeated query contexts can actually identify under context permutations,
and distinguish intrinsic pretext ambiguity from a particular sampled context
assignment. Neither that proposed check nor this prefix-only diagnostic is an
ICLR-ready general causal explanation of transfer.

Worktree/branch: `/Users/philipp/projects/gfm/prodigy-mechanisms`,
`codex/target-performance-mechanisms`; dedicated Tucker worktree
`/dataMeR1/phil/gfm/prodigy-mechanisms-role`. No GPU work or production-default
change was made in this diagnostic.
