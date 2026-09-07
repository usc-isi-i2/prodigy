# Fixed-routing reference transport is affine, not an unidentified nonlinearity

7 September 2026. Research calculation from the implemented metagraph and one
already-saved episode; no model forwards or new intervention outcomes.

## Exact computation

For the audited single-M, evaluation-BatchNorm model, let V be the value
projection weight, O the output projection weight, D the diagonal saved BN
scale, and A(cs) the diagonal matrix repeating each head's actual attention
weight for the support-s to class-c edge over that head's channels. With
routing fixed, a change in support embedding delta z(s) gives

`delta L(c) = sum_s D O A(cs) V delta z(s)`.

This is an affine-path difference: value/output biases, unchanged label
residuals and self-messages, and the shared BN offset cancel. The expression
must not move A through V or O; head-dependent routing generally prevents
replacing this with one attention-independent matrix. It assumes the exact
single-layer/no-final-back/evaluation-BN contract, not the public M2 runtime.

The final normalized cosine readout still depends on reference lengths and
directions. Thus an affine change of unnormalized references can produce
non-additive norm/direction performance effects. Such non-additivity is not
evidence of an unexplained upstream nonlinear operating range.

## Saved-tensor check

Used the first indexed original 50k episode, `batch_000.pt`, not selected by
performance. Multiplied the saved intact attention, checkpoint V/O/BN scale,
and removed-minus-intact support embeddings. The resulting class-reference
change matches saved values-only-minus-intact final references with maximum
absolute error 1.598e-6 (maximum actual change 5.025), relative L2 error
1.986e-7. This is a float64 reconstruction of float32 operations, not a
bit-exact claim. Saved attention and query representations are exactly equal
between intact and values-only endpoints.

Input SHA256: `b5591f5a79bcccba9495b53569d071ec4471059cc8f5828f892a4e3a50b7a78a`.
Tucker input: `prodigy-classrefkv/log/classrefkv_long_20260907/private_activations/original/batch_000.pt`.
Weights: nominated HK50k `cp_hk_twitter_nm_bio_02_07_2026_08_58_57`,
`layer_list.2.gnn_layers.0`; saved BN epsilon is the implementation default 1e-5.

## Hypothesis and guard against an invalid simplification

The remaining candidate is incompatibility between this learned transport
and the target support directions that should define class references. This
would be more specific than saying supports have large norms or that values
matter. It remains a hypothesis, not established by the reconstruction.

Do not use `(I + OV)^T D^2 OV` as the actual query-support kernel: queries
receive label-node messages as well as self-messages, and the head weights
vary by receiving node. A spectrum of OV alone is not an invariant explanation
of the observed classification effect. A useful property must account for
the receiving query representation and the exercised support directions,
without merely re-expressing target AUC as an alignment statistic.

The next mechanism decision must distinguish transport incompatibility from
bad support directions already supplied by the encoder. A raw-versus-learned
readout gap does not isolate that distinction because it changes several
computations. This note supplies the correct operator for that question; it
does not nominate another sweep or claim that the acceptance-critical
predictive contrast has been established.

Private worktree `.worktrees/role-topology`, branch
`codex/role-topology-interactions`, code `30df8cb5`. No manuscript change.
