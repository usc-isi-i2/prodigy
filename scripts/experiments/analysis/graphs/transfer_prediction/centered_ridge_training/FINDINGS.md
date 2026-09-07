# Strengthened direct training closes the representation-utility gap

The predeclared comparison rejects the interpretation that the earlier isolation
pilot established a special benefit from native-inference training. Training a
support-centered ridge objective with a learned positive scale closes the earlier
gap to native-trained encoders under the same frozen centered-U1 deployment rule.
This is one training initialization, two schedules, and two fixed episode streams;
the streams are not independent training seeds.

## Fresh stream, previously fixed supported-four panel

Values are percentages; every row uses the same centered-U1 deployment readout.

| Training | Schedule | Accuracy | Macro-F1 | AUC |
|---|---|---:|---:|---:|
| Native | Blocked | 80.876 | 80.245 | 86.085 |
| Strengthened direct ridge | Blocked | 81.673 | 81.092 | 86.017 |
| Native | Interleaved | 80.656 | 80.079 | 84.400 |
| Strengthened direct ridge | Interleaved | 81.673 | 81.117 | 85.710 |

Accuracy improves by 0.798/1.017 points over native-trained U1, and macro-F1
by 0.846/1.038 points. Blocked AUC is slightly lower (0.068 points), so this is
not uniform dominance across metrics. Original-stream direct accuracy is
81.852/81.706, with macro-F1 81.209/81.096. All five targets, including the
previously unsupported suspended-account target, remain in the full export.

## Interpretation and decision

The old direct objective was an inadequate basis for claiming that the native
solver supplied irreplaceable representation-training benefits. Centering and
learned scale changed together; this experiment cannot attribute the improvement
to either alone. It establishes neither universal equivalence nor a novel
auxiliary-head principle. The public temporal crossover findings remain separate.

The positive lead is a possible training/deployment simplification: direct
optimization of a support-fitted readout can match or improve these representations
without native-loss gradients. No compute benefit has yet been demonstrated:
this compatibility implementation still executes the detached native forward.
Any efficiency claim requires actually removing that work and measuring quality
against compute, not just counting trainable modules.

## Provenance and checks

Training runtime fc9a0396; evaluation runtime 7e111c20. Both 2,500-update arms
match original native initialization exactly and match all 625 consumed payloads
per source (IDs, roles, topology; not every feature byte). Learned training scales
are 36.4286 blocked and 22.0144 interleaved; deployment scale remains one.
Cached native predictions, intermediate embeddings, and original ridge predictions
pass parity checks on the first batch of every target/stream; all batch hashes
are checked and evaluation must leave model state unchanged.

The local data/centered_ridge_eval_20260907 export contains 80 metric rows,
32 panel means, and receipts for 640 batch predictions. Canonical predictions:
/dataMeR1/phil/gfm/prodigy-centered-ridge-eval/log/centered_ridge_eval_20260907.
Native comparison source: public_prodigy_kg/data/social_centered_readout_20260907
in the codex/publickg-paired-state analysis worktree.

Independent post-run audit verified all 640 saved output hashes and corresponding
source-file hashes and recomputed all 32 panel means to 1e-12. Summary SHA256:
ec6339f8fa96e0a0df0b1e537894a48993cdb75207aa44bcdfd69abdc557284c.

### Earlier smoke validation

Both 20-update arms completed at 1fa5a048 before the full run, with exact
initialization and five matched payloads per source audited at fc9a0396.
Positive scales changed to 14.7665567/14.6168947. Smoke checkpoint SHA256:
blocked 2fe34e2c233dbf319e83558a13da13d3225c6c9db20abf91120077f5b35bcb4c;
interleaved f013832ebd3c6e9bc0c10230219b2c19e5265545f4344ad6d9e795b85540290e.
Smoke artifacts remain in the training worktree at log/centered_ridge_training_smoke.
The training progress-bar native loss is an unused-head diagnostic; optimized
loss and scale are recorded as train_ridge_loss and train_ridge_logit_scale.
