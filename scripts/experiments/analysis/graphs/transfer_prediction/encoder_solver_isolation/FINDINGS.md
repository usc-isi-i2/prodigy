# Encoder–solver isolation: completed pilot, AUC no-go

## Completed pilot: no-go for this isolation protocol

All eight 2500-update arms and all ten target/stream replays completed at runtime
revision `0b9d9641`. Full audit confirmed eight exactly matching initial model
states and exact per-source consumed-episode payloads across all arms. Checkpoint
encoder states are not identical across isolated/ridge-only trajectories.

Read-only aggregation of production `metrics.jsonl` under Tucker
`log/isolation_eval_20260907` gives these equal-weight fixed-four-target results
(Election2020, TwiBot20, political, Facebook; suspended excluded by prior design):

| Stream / schedule | Native full AUC | Joint full AUC | Isolated full AUC | Isolated U1 AUC | Joint U1 AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original / blocked | .831227 | .807824 | .799227 | .804072 | .845924 |
| Original / interleaved | .823950 | .824850 | .799132 | .813763 | .854204 |
| Fresh / blocked | .830391 | .812589 | .801157 | .808179 | .850818 |
| Fresh / interleaved | .816190 | .831306 | .802342 | .811797 | .854919 |

Isolation loses to native full inference and its own U1 readout on AUC in all
four comparisons. This fails the advance criterion; do not expand this exact
protocol to three seeds or claim a successful repair. These are one training
seed and reused episode streams, not four independent replications. They do not
prove gradient isolation can never help under another loss scale or protocol.

Joint training's U1 AUC exceeds isolated U1 by 4.04–4.31 points across these
comparisons. The intervention does not support the simple story that learned
solver gradients are uniformly harmful to transferable features. A distinct
hypothesis is that native supervision helps features while deployment of the
learned solver remains suboptimal. This is interpretation, not causal mediation
or evidence of a competitive new method.

### Target-level qualification

Production per-target AUC makes the limitation more informative than the panel
mean alone. Under blocked training, isolation raises TwiBot20 full AUC from
.60450 to .68567 (original) and .59107 to .66386 (fresh), but lowers Facebook
from .79779 to .64759 and .79910 to .66603, respectively. Political full AUC
also falls by about five to six points. This is a target-dependent trade-off,
not uniform damage from isolation or universal benefit from solver gradients.

Adding ridge to native training also separates feature utility from deployed
inference: on blocked TwiBot20, joint U1 AUC rises from .66707 to .67333
(original) and .65253 to .67228 (fresh), while full AUC falls from .60450 to
.55866 and .59107 to .55649. These paired outcomes corroborate the need to
evaluate both stages, but do not independently prove why the separation occurs.
The two streams reuse one set of trained checkpoints.

Macro-F1 analysis is pending deployment of local revision `d5843502`: the first
analysis failed its parity guard because mapping local argmax decisions is not
equivalent to taking argmax after global probability mapping at ties. Production
metrics are unchanged. Corrected read-only calculations exactly reproduced the
previously failing suspended/raw-ridge and TwiBot20/isolated-U1 cells; three local
tests pass. Full corrected analysis must still pass before reporting macro-F1.
GitHub push authorization remains pending; do not bypass that restriction.

## 2026-09-07 blocked-schedule smoke

Runtime revision `295723f8`, Tucker output
`/dataMeR1/phil/gfm/prodigy-encoder-solver-isolation/log/isolation_smoke_20260907`.
All four blocked arms completed 20 updates. Their consumed-episode payload
digests match exactly using the existing audit (IDs, context topology and roles;
this is not an independent hash of every input feature tensor).

The expected isolated/ridge-only encoder equality does **not** hold in these
checkpoints: across 19 encoder tensors the maximum absolute difference is
0.1837847605 (first background-layer BN running mean). First MLP weight maximum
difference is 0.0382952522. Optimizer states both report 20 updates; objective
metadata matches the requested modes. Unused learned label embeddings and
unused reset-MLP tensors match exactly, consistent with—but not proof of—matched
initialization. This legacy smoke did not save step-zero weights.

The CPU model tests use a simplified encoder and do not establish numerical
parity of the actual CUDA scatter-based encoder. A real-batch paired-gradient
check with an identical-objective null is required to distinguish numerical
variation from an unintended gradient path. Do not silently loosen an error
tolerance to approve the run.

## Real-batch CUDA null comparison

Diagnostic revision `0b9d9641`, output
`/dataMeR1/phil/gfm/prodigy-isolation-parity/log/parity_20260907.json`.
Actual first training batch on GPU1; all input tensor hashes and restored model
states match exactly across three forwards. No added deterministic settings.

Isolated versus identical isolated repeat: maximum encoder gradient difference
6.333e-8; post-AdamW encoder-state difference 0.0012785; maximum U1/full-output
difference 4.292e-6. Isolated versus ridge-only: corresponding differences
5.960e-8, 0.0011033, and 4.768e-6. All three ridge losses are 3.3591527939.
The largest one-step parameter differences are in `lin_self_loops.bias` and
`mlp.2.bias`, consistent with AdamW amplifying tiny reduction differences.

Decision: the one-batch isolation comparison shows no excess gradient error
relative to the identical-objective null. Exact CUDA trajectory equality is not
a valid smoke criterion here. This does not prove that every 20-step difference
is numerical or establish multi-seed robustness. Proceed with the fixed pilot
after all smoke arms and consumed-example audits pass; capture initial weights,
retain measured parity errors, and expand seeds only for a promising outcome.

## Outcome analysis prepared

`analyze.py` compares full inference, U1 ridge, and raw-center ridge on all five
fixed targets and both episode streams. It reports accuracy, macro F1,
positive/local-class F1, AUC and NLL. Five-target and fixed four-target means use
equal target weights; streams remain separate. On Facebook, binary F1 uses
episode-local classes, not a semantic positive category across the dataset.
