# Implementation validation only

The paired toy run (two conditions, three updates) and full-graph smoke (three
sources, three seeds, two conditions, eight updates) ran on Tucker CPUs at
`c91e2aaf`. The completed full-graph states and inputs were independently rechecked
with the strengthened verifier at `f20e6495`, after the smoke session exited.
These are not target-performance results and must not enter substantive tables.

The substantive 18-model experiment started at 2026-09-06 11:01:24 UTC in
`/dataMeR1/phil/gfm/prodigy-mechanisms-freeze`, frozen revision `f20e6495`, tmux
`mechanism-readout-training`. Its output is
`log/target_mechanisms/readout_constraint_training_20260906`; automatic evaluation
is gated on full training verification. Smoke results do not establish that this
substantive experiment has completed or that the predicted remedy works.
