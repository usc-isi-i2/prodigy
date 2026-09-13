# Complete source-pair composition panel

This experiment tests whether the source-composition result inferred from three
nested ladders holds on non-nested mixtures. It trains every unordered pair of
the nine registered final-core sources under the same 2,500-update, balanced,
interleaved, graph-local PRODIGY protocol.

The exact design is 36 source pairs by three training seeds, giving 108 physical
models. Every terminal checkpoint is evaluated on the same nine fixed 512-episode
neighbor-matching panels, for 972 audited cells. The training seed is the
replication unit. Fixed episodes pair models within a target and do not provide
additional independent seeds.

The complete pair panel permits three checks that the nested ladders cannot:

- compare source composition while holding source count fixed at two;
- test the best-included-specialist envelope on non-nested source sets;
- estimate which donors complement one another without selecting an order from
  the observed receiver matrix.

`run_tucker.sh` accepts `PHASE=train` and `PHASE=eval`. In production,
`wait_and_run_tucker.sh` starts the 108-model training phase ten per GPU after
mechanism training while the long fixed-exposure tail finishes. At the live
handoff only 13 primary jobs remained, so this keeps total trainer concurrency
near the already measured 14-per-device envelope while reducing the pair panel
from six waves to four. The flagship recovery waits for this bounded phase if
the old remainder exits early. Pair evaluation remains behind the flagship
native/downstream results, optimized core audit, and completed mechanism audit;
it acquires only Tucker GPUs 0--3 after a stability check and evaluates all 972
cells. An existing output root is never overwritten.

The production defaults use 14 models per GPU and a total 224-loader-worker
budget. This is the same per-device concurrency already measured while the
two-hop remainder and mechanism campaigns shared GPUs 0, 2, and 3: about
24 GiB on each 80 GiB GPU, with four loader workers per model and substantial
host-memory headroom. For 108 pair models this changes the shared-graph training
from four waves at the earlier eight-per-GPU setting to two waves. Both values
remain explicit environment overrides.
