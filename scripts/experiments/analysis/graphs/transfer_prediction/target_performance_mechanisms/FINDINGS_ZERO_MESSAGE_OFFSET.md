# Zero messages expose a large learned affine offset

7 September 2026. Diagnostic localization, not a new performance experiment.
Private; do not push with public code.

## Exact computation

Both inspected checkpoints have identity input projection, 768-dimensional
inputs and a 256-dimensional S output. Thus the dimension-conditional residual
in SAGEConvSelfLoops is **absent**, while the learned self projection remains.
There is one background layer, no intermediate reset, and no outer skip path.
With support messages zeroed, the pre-BN center vector is exactly

`z(x) = W_self x + b_self + MLP(0)`.

Deleting messages therefore does not remove all feature dependence. It leaves
a learned feature map plus a shared offset. MLP(0) need not be small.

## First retained original batch: localization

Read-only CPU replay using the existing forward_scaled helper and load_cell,
code 9e19e5eb on Tucker's prodigy-label-context-discovery worktree. No files,
weights, or training jobs changed. Four endpoint forwards: intact/suppressed
for each checkpoint. These are first-batch diagnostics, not full-panel estimates.
Early batch contains 80 support centers across four episodes; late contains
six across one episode. Dispersion below is across that batch's support centers,
not an equal-episode population estimate.

| Quantity | Early 2.5k | Late 50k |
|---|---:|---:|
| Norm of MLP(0) | 2.192695 | 640.434998 |
| Norm of self bias | 1.531684 | .341042 |
| Norm of combined shared offset | 3.468470 | 640.457947 |
| Mean norm of feature-dependent W_self x | 2.899015 | 39.785435 |
| Centered RMS of feature-dependent vectors | 1.991267 | 16.491882 |
| Unit dispersion of feature-dependent vectors | .467585 | .129748 |
| Unit dispersion after adding offset / pre-BN | .184085 | .000643 |
| Suppressed post-BN unit dispersion | .534534 | .000767 |
| Suppressed post-ReLU unit dispersion | .482639 | .000799 |
| Suppressed U1 unit dispersion | .158168 | .000290 |

The independently computed affine sum reproduces the replay's pre-BN unit
dispersion (.000642789035 vs .000642789011 late). This scalar agreement is
not a tensor-level reconstruction audit. The first decomposition attempt
incorrectly assumed a residual and failed on 768 versus 256 dimensions;
the reported calculation follows the actual conditional branch.

## What this explains, and what it does not

The late first-batch angular collapse already exists before BN/ReLU. The
feature-dependent component remains diverse; adding the learned shared offset
reduces its unit dispersion about 202-fold. The dominant term is MLP(0),
not the self-projection bias. This supplies a concrete competing explanation
to "normalization creates collapse": suppression exposes a large nonzero
default output of the message MLP, overwhelming the surviving feature signal.
BN/ReLU can still affect the final result; they are not exonerated globally.

This explains the location and arithmetic of the observed first-batch collapse,
not yet the downstream AUC change. Earlier fixed-attention common-shift tests
acted at a different stage and cannot substitute for this encoder diagnosis.
Removing this offset has not been tested as a correction. Nor does this show
training duration causes the large offset: the checkpoints differ in protocol.

Next bounded check: reproduce this decomposition across the already retained
episodes, with tensor-level endpoint agreement, before considering a single
offset intervention. Do not launch a new checkpoint or target sweep. Preserve
the distinction between early robust repair and late initialization-sensitive
ranking improvement.

## Completed full retained-episode verification

All 32 early original batches (128 episodes), 128 late original batches and
128 late fresh batches were replayed with zero support messages on CPU.
Every input batch hash matched its original receipt. Concatenated logits
were bit-exact with the saved suppressed endpoint in each configuration/stream.
No new target examples, training, or fitted parameters were introduced.

| Equal-episode mean | Early original | Late original | Late fresh |
|---|---:|---:|---:|
| Feature-dependent unit dispersion | .468198 | .184641 | .189784 |
| Pre-BN dispersion after shared offset | .185119 | .000775 | .000806 |
| Post-BN dispersion | .540987 | .000725 | .000742 |
| Post-ReLU dispersion | .484957 | .000737 | .000756 |
| U1 dispersion | .153734 | .000272 | .000285 |
| Mean feature-dependent norm | 2.911847 | 40.184547 | 40.488660 |

Adding the offset reduced dispersion in every episode. Ratios of the late
mean feature-dependent to pre-BN dispersions are approximately 238.3 and
235.4. This is a ratio of means, not the mean of episode-wise ratios.

The explicit affine sum was compared tensor-by-tensor against actual pre-BN
support-center activations captured by forward_scaled. Maximum absolute
reconstruction error was 1.19e-7 early and 3.81e-5 late; maximum error relative
to each batch's peak absolute activation was 1.09e-7 early and 3.20e-7 late.
All passed atol=1e-4, rtol=1e-5. The reconstruction is numerically equivalent,
not bit-exact; the full saved endpoint logits were independently bit-exact.

This completes the planned verification. The evidence supports a learned
affine explanation of **suppression-induced collapse**. It does not establish
that this offset causes the original intact model's transfer failure, or that
subtracting it improves discrimination. Those are distinct hypotheses.

## Advisor decision after independent review: one paired test

Before any offset-subtraction outcomes, nominate subtraction of exactly
`MLP(0)` at the support-side pre-BN input, applied to BOTH intact and suppressed
contexts. Do not subtract the self bias, tune a multiplier, change the label
initialization, or search another checkpoint. Use the already nominated late
checkpoint and its original/fresh retained streams. Existing native endpoints
complete the 2-by-2 design. This test has not yet run.

Primary mechanistic prediction: subtraction increases suppressed support
angular dispersion before BN in each stream. This is not a prediction that
AUC must improve. Record U1/reference geometry, within-episode and pooled AUC,
accuracy, and exact query preservation. Verify the subtraction against the
parameter-derived vector and preserve original input/weight identities.

Interpretation fixed before outcomes:

- Only suppressed geometry recovers: explanation of a suppression side effect,
  not evidence that native context processing is defective.
- Geometry recovers but reference/decision pathology remains: upstream collapse
  is explained, not the full downstream failure.
- Intact ranking and accuracy also improve: evidence that the operating point
  matters for ordinary inference, requiring scoped interpretation rather than
  a universal correction claim.
- Intact performance worsens: the offset can be useful or compensated in the
  natural operating regime; a large norm alone is not a defect.

The independent reviewer recommended this paired design specifically to avoid
mistaking repair of an artificial ablation for explanation of native transfer.
