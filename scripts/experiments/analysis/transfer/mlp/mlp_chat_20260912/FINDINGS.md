# MLP ladder chat closeout — 2026-09-12

## Preserved work and evidence

- Metrics/checkpoint implementation: `53de530`, artifact tests `7f8ec81`; already integrated into the local main checkout. Training and validation scalar reports, raw evaluation logits, labels and calibration diagnostics, atomic best/latest/periodic/terminal state. Offline W&B remains default; independent simultaneous runs. Exact sampler resume is NOT implemented.
- Validation at implementation time: 15 targeted tests passed; broader suite 67 passed / 1 unrelated existing PyG `Data.keys()` compatibility failure. Tiny CPU training exercised periodic and terminal checkpoint persistence. Rung-9 Facebook smoke evaluation reproduced its stored BCE/accuracy from saved scores; selected step 859500, cosine and dot metrics distinguished. This smoke is archived in `data/evaluation_results.tar.gz`.
- Earlier ladder histories, activation/weight and distribution investigations, corrupted Suspended finding, and fast-loader evidence already committed under `results/mlp_ladder_diagnostics_20260911/` (`d91984b`, reachable here). Original static-LP results use the pair evaluator, not obsolete episodic LP.
- Missing loose diagnostics now archived under `data/` and `figures/`: raw-dot AUC/BCE ladder, dimension ablation, untrained BCE baseline, input-coordinate/PCA pilot comparison, and requested plots.
- Transfer matrix versus ladder: `transfer_comparison/FINDINGS.md`, scripts, exact inputs, cell comparisons and both figure layouts. Full historical best-specialist predictor reduces squared gain error 52%; pre-Suspended rungs 65%. Much weaker for still-unseen targets. Oracle chosen retrospectively per target, with unmatched specialist/ladder budgets: not causal or independent predictive evidence.
- Input study producer is `337a303` (equivalent integrated cherry-pick `fb5d0fa`); offline report repair is preserved separately on `codex/mlp-input-study-report` at `1dc3241`. Earlier inspection scripts remain on their remote diagnostic branches. No branch is deleted in this closeout.

## Conversation conclusions and untested hypotheses

Raw input-coordinate selection retained less predictive signal than PCA or the full input in the fixed 90k-step pilot. This is not a mutual-information estimate or convergence comparison. Hidden-unit compressibility and sparse activations do not establish a narrow set of informative raw coordinates. BCE must be interpreted with negative ratio and score calibration: a constant p=1/6 gives approximately 0.45056 BCE for 1:5 pairs; balanced evaluation has a 0.69315 constant baseline. Constant-score AUC is 0.5.

Larger hidden widths (256/512/1024 while retaining output 256) were proposed but NOT run here. Equal round-robin updates need not imply equal progress: unequal graph sizes/difficulty, overfitting, undertraining and cross-source interference are possible. They were not established by the aggregate plot. Proposed next diagnostics are per-source training/validation trajectories and controlled width or scheduling interventions. Training longer until every source plateaus can continue updating saturated sources. Use repaired data and matched protocols before claiming remedies work.

## Retention and deletion scope

`data/histories.tar.gz` contains 89 configuration, summary, JSONL history and tracking-receipt files from the historical ladder and input pilot. `data/evaluation_results.tar.gz` preserves 81 ladder evaluations, aggregate tables, input-study aggregate results, and the new smoke evaluation including its NPZ predictions. `data/tucker_inventory.json` records 129 files and SHA256 hashes for 40 retained checkpoint files across ladder and input-study state roots. Large checkpoints and original offline W&B binaries remain on Tucker, outside the metrics worktree; they are not Git blobs. Fixed-budget historical figures/histories are in the earlier diagnostics archive.

A broader concurrent retention archive also exists on `codex/experiment-retention-audit` (`79b8d79`), including copies of older diagnostic reports and the originally loose comparison directory. This focused archive makes this chat's evidence discoverable from main without relying on that branch.

Important unresolved external location: the old specialist path `/dataMeR1/phil/gfm/mixture-scaling-node-only/state/node_only_transfer` no longer exists at this audit. The specialist result table is preserved here and in commits `e61153d`/`f54b7d8`; its original checkpoint location could not be verified. This closeout did not remove that worktree or its files and does not claim those checkpoints are retained. All four ladder/input-study roots were verified present.

Only `/tmp/mlp-metrics-checkpoints` and Tucker `/dataMeR1/phil/gfm/mixture-scaling-metrics-checkpoints` are approved for removal here. Both have committed code plus disposable Python/pytest caches, no experiment state, and no active job at audit. Main checkouts, all other experiment worktrees, source datasets, large run state, and branches are retained.

See `data/manifest.json` for hashes of archived evidence. Archives contain data only, not credentials. Historical Suspended-dependent results remain suspect and retain their caveats; archiving is not validation of their scientific conclusions.
