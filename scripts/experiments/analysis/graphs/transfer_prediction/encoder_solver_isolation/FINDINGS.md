# Encoder–solver isolation: implementation audit in progress

No target-transfer outcome is available yet. Do not interpret smoke training as
evidence for the proposed training protocol.

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
variation from an unintended gradient path. Full training is on hold pending
that diagnosis. Do not silently loosen an error tolerance to approve the run.

## Outcome analysis prepared

`analyze.py` compares full inference, U1 ridge, and raw-center ridge on all five
fixed targets and both episode streams. It reports accuracy, macro F1,
positive/local-class F1, AUC and NLL. Five-target and fixed four-target means use
equal target weights; streams remain separate. On Facebook, binary F1 uses
episode-local classes, not a semantic positive category across the dataset.
