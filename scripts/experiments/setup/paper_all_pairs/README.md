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

`wait_and_run_tucker.sh` waits for the flagship training, flagship native and
downstream evaluations, and the optimized core evaluation to pass their exact
coverage gates. It then acquires only Tucker GPUs 0--3 after a stability check,
trains the shared-graph batch, and evaluates all 972 cells. An existing output
root is never overwritten.
