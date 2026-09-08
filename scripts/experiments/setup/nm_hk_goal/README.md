# HK NM mechanism and bounded repair

Authorized overnight goal, 8 September 2026. HK→HK NM first; source controls
remain part of interpretation. No classification work. Existing jobs, including
the queued flagship launcher waiting for all GPUs, have priority. Begin with
two CPU threads and no CUDA allocation. Use only an idle owned GPU (0–3) after
the priority queue clears; never alter or stop another process.

## Frozen first question

On the existing 200 selected cases, compare the original and nearest-support
endpoints using the same cached HK embeddings. Substitute only the three
changed supports' actual metagraph keys, values, or both. Retain every other
row/block, the original native decoder, and evaluation normalization. All Q
blocks remain unchanged. Query labels select metrics, never the forward pass.

The score-only analysis has already shown that true-class score movement is
sufficient to reproduce all 200 nearest-support correctness outcomes. The next
question is whether that movement follows changed attention, changed values,
or their interaction. No K/V outcome from these NM inputs has yet been seen.

Required checks: exact unchanged query outputs; joint K/V matches the full
replacement's final label/query outputs (1e-5 tensor, 1e-4 logit tolerances);
value-only attention identical to original; keys-only attention identical to
replacement; original/replacement predictions and probabilities match cached
references (1e-5 probability, 1e-4 tied-winner logit tolerances). Count numerical
differences, do not silently loosen gates. Preserve weights, inputs and donors.

Report both failed cases and correct controls. Five random trials remain
correlated draws if used later. K/V results on prior classification experiments
are not NM evidence. The 71 persistent cases are a diagnostic subset, not a
benchmark denominator.

After this first result, record the repair rationale and its fixed evaluation
protocol before testing it. Prefer a small, source-trained or source-validation
selected change; never fit to the known canonical test labels. Include unchanged
full canonical accuracy, recoveries and breakages, and source controls. A failed
repair remains a valid result of a bounded test; it must not be reported as a
successful remedy.

Initial setup estimate: 30–60 minutes. CPU compute estimate: 5–15 minutes.
New graph loading, sampling and encoding are unnecessary for this first stage.

## Narrow follow-up selected after the first K/V result

The completed K/V test (`af8d8596`, CPU, 50.18 seconds) found that value-only
reproduces all 25 nearest rescues and 51/54 nearest breakages. Key-only reproduces
none of those rescues and 19/54 breakages. Before choosing a repair, separate
actual projected support-value radius and direction, holding original attention
fixed. For each of the three ordered support slots, use replacement direction
with original radius, or original direction with replacement radius. Preserve
all other rows and blocks. These are diagnostic donor interventions; original
support radii are not available to a general replacement policy.

No additive causal percentage or proof of nonlinearity follows from an
interaction between these hybrids. Even an affine map admits a radial/directional
interaction. This one additional distinction guides a bounded repair; it is not
an open-ended component sweep. Setup estimate 10–20 minutes, CPU compute 1–2
minutes, no GPU. The first failed start stopped before inference because the
effective config path was wrong; its log is retained and excluded from results.

## Frozen bounded repair, after the radius/direction result

Value direction alone reproduces 21/25 nearest rescues and 50/54 breakages;
radius alone reproduces 2/25 and 13/54. Direction-only also reproduces all five
nearest rescues in the 71 persistent cases. This motivates preserving explicit
query/support geometry around the learned class-reference construction. It does
not show that native values should be discarded: the original native head is
better than the direct cosine head on the full HK stream.

Test exactly one parameter-free inference change: for each query, center each
head's 30 class scores and divide by its population standard deviation (floor
1e-8), then average the native head and mean pre-metagraph support cosine with
equal weights. This is an explicit symmetric default, not an optimized weight.
No query labels enter either head; only observed support labels define classes.
No fitting, temperature selection, target-specific choice, or weight sweep.
Standardized outputs are ranking scores, not calibrated probabilities. They
are invariant to each head's positive scale and offset, except at the floor.
The existing support-prototype flag changes label inputs before M, while the
learned relation scorer operates after M; neither preserves this direct route.

Primary: original, unchanged full canonical HK target (61,440 occurrences),
native HK checkpoint. Controls: Ukraine checkpoint on the same HK inputs, and
both checkpoints on the original Ukraine target. Report native/direct/residual
accuracy, recoveries and losses relative to the original canonical predictions,
node-weighted accuracy, query-frequency strata, and episode-level paired deltas.
The 512 episodes and repeated nodes do not constitute independent model seeds.
All four cells use existing hashed caches. Recompute the geometry score from
support signs and check it against cached cosine scores; do not re-encode graphs.

Secondary: the already selected original and replacement HK cases, separately
by method and cohort. Report nearest and random draws without picking a winning
method. These remain known-true-class interventions, not benchmark performance.
Report differences from both the original native outcomes and the native head
on the SAME modified inputs. One failed repair is a completed bounded test;
do not tune on the canonical test to reverse an inconvenient result.

Setup estimate: 20–35 minutes. CPU estimate: 2–5 minutes, two low-priority
threads, no CUDA allocation. The optional production flag is off by default,
NM-only and inference-only, and adds no checkpoint parameters. It is retained
as an experimental implementation regardless of the outcome, not recommended
as a successful model improvement before evaluation.

Runtime branch: `codex/nm-hk-goal-20260908`; local worktree
`/private/tmp/prodigy-nm-hk-goal`. Transfer source via private git and use a new
Tucker worktree. All paths, methods and thread counts must be explicit/overridable.
