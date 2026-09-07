# Suppression does not uniformly improve support geometry

7 September 2026. Saved-tensor diagnostic specified before reading its new
outcomes: compare intact and suppressed supports using the same intact U1
queries and a fixed simple prototype, then compare the direction with native
inference. Zero model forwards, no fitted parameters, no new episode stream.
The episodes and native intervention outcomes were already known; this is not
a pristine held-out generality test.

## Computation and checks

For each class, average individually L2-normalized pre-metagraph support
embeddings, then normalize that class mean. Score the fixed intact normalized
U1 query embeddings by cosine against those means. Compare intact versus
support-suppressed support embeddings. Query U1 tensors are bit-exact across
the two endpoints. This is a particular prototype readout, not all possible
tests of representation quality or ridge.

Recover class membership from saved positive support-to-label edges; recover
episode query membership from label-to-query edges. This respects the four
episodes per discovery batch and one per 50k batch. Check nonempty support
sets, no query in support sets, both query classes, complete query count, and
matching saved local query labels. Average local margin AUC within episodes,
then equally across 128 episodes. Native deltas reconstructed from the same
saved logits reproduce the previous K/V report.

| Checkpoint / stream | Intact prototype AUC | Suppressed prototype AUC | Prototype change | Native inference change |
|---|---:|---:|---:|---:|
| 2.5k / original | .880136 | .913267 | +.033131 | +.072483 |
| 50k / original | .621383 | .556858 | -.064525 | +.087637 |
| 50k / fresh | .635851 | .572338 | -.063513 | +.103516 |

Each row has 128 episodes and 3,072 query occurrences. Two streams are not
training seeds, and the 2.5k and 50k configurations differ in protocol as
previously documented. No confidence or significance claim is attached.

## Decision and explanatory consequence

The 50k result contradicts the simple account that removing support context
helps because it uniformly improves the class geometry delivered by the
encoder. Under a fixed simple readout and fixed queries, that geometry becomes
less useful, while the native constructor improves discrimination. Its benefit
is conditional on downstream processing. At 2.5k, prototypes improve too, so
the early result does not isolate constructor-specific incompatibility.

Combined with the existing fixed-query/fixed-attention value intervention,
this supports investigating learned transport compatibility at 50k. It does
not isolate the V matrix: the prototype comparison also changes query-space
processing, class weighting, affine offsets and final normalization. Nor does
it prove the encoder generally produces good representations; the 50k intact
prototype is only about .63 AUC. A cross-source predictive property remains
unestablished. Preserve the separate result that native ranking improvement
does not repair threshold accuracy.

Do not generalize the early/late difference into a training-time law: these
are different configurations, not a controlled training trajectory. Do not
discard the early result because the late contrast makes a sharper story.

## Evidence

Tucker root `prodigy-classrefkv/log`; runs
`classrefkv_discovery_20260907` and `classrefkv_long_20260907`.
Read `private_activations/<stream>/batch_*.pt` in sorted order and
`private_predictions/<stream>.pt`. SHA256 of newline-joined SHA256 values of
the sorted activation files:

- Discovery original: `4b394aebbefc2f53f7927a4db215ba5328d2fdc5535ad822d72aa4fbf85915ed`
- Long original: `2886948253ea3a91f017d3bc228fb63e4266ba769c82d30b1c91bd16fee2175c`
- Long fresh: `72f7672a94ece638e3f5117d2c54404fbea7c5ac5adeafe642a2663c3ef61a31`

These record the files read, not comparison against a prior immutable manifest.
Private worktree `.worktrees/role-topology`, branch
`codex/role-topology-interactions`, code `30df8cb5`. No manuscript expansion.
