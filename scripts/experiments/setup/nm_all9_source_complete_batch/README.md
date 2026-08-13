# All-nine source-complete large batch

This diagnostic asks whether the all-nine PRODIGY mixture changes when every optimizer
update averages gradients from all nine sources instead of seeing one randomly selected
source episode at a time.

Each batch contains exactly nine neighbor-matching episodes:

- one episode from each `graph_id` in the all-nine/Facebook merge;
- every episode is internally confined to its source, including all centers, positives,
  and negatives;
- source order within the batch is shuffled;
- the ordinary mean loss and the historical `1e-3` learning rate are retained.

This is not obtained by merely setting `batch_size: 9`. Independent balanced draws would
contain only about 5.9 distinct sources on average, and would contain every source exactly
once with probability `9! / 9^9 = 0.000936`.

## Comparisons

The full config saves two especially important checkpoints:

- step **4,444**: 39,996 episodes total, approximately exposure-matched to the historical
  batch-1 40k checkpoint, but with 4,444 optimizer updates;
- step **40,000**: 360,000 episodes total, 40k per source, matching the historical
  optimizer-step count but using nine times the episode exposure.

The two checkpoints separate the most useful readings available from one run. Neither is
individually a perfectly controlled large-batch comparison because changing batch size
necessarily trades off update count against sample exposure.

## Run on Tucker

Use the `prodigy` environment and an owned GPU. Start with the smoke config:

```bash
DEVICE=0 bash scripts/experiments/setup/nm_all9_source_complete_batch/train_tucker.sh \
  scripts/experiments/setup/nm_all9_source_complete_batch/smoke.yaml
```

If memory and throughput are acceptable, launch the trajectory:

```bash
DEVICE=0 bash scripts/experiments/setup/nm_all9_source_complete_batch/train_tucker.sh \
  scripts/experiments/setup/nm_all9_source_complete_batch/train.yaml
```

The required merged artifact is
`/dataMeR1/phil/data/merged/graphs/ukr_rus_covid_midterm_all9_facebook_graph.pt`.

## Status and result

The full run and the 18 native-NM evaluations completed on 2026-08-07. At matched
episode exposure, batch 9 averaged `0.3939` accuracy versus `0.4179` for the
historical batch-1 model (`-0.0240`). At 40k optimizer steps, after nine times the
episode exposure, batch 9 reached `0.4195` (`+0.0016`). The large batch therefore did
not improve sample efficiency and is not recommended as the default.

See `scripts/experiments/analysis/transfer/ablations/batch_construction/nm_all9_source_complete_batch/FINDINGS.md` for the
per-dataset comparison and saved-run paths.
