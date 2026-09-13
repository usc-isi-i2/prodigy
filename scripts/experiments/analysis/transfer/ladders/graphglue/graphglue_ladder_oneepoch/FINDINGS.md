# GraphGlue one-epoch source-addition ladder

Updated September 12, 2026. All seven models and 70 target evaluations completed. **Provisional findings for this one-epoch configuration; not a successful reproduction or refutation of Figure 5.** This report covers the later one-seed/full-epoch experiment, not the earlier three-seed capped pilot in [FINAL_REPORT.md](/Users/philipp/projects/gfm/GraphGlue-results/FINAL_REPORT.md).

## Results

Test accuracy is the mean across ten fixed target splits. SD is population standard deviation across these splits, not across pretraining seeds or a confidence interval. Changes are paired means on matching split IDs. Split counts describe this fixed panel; the splits overlap, so they are not ten independent datasets.

| Path | Rung | Test accuracy ± SD (%) | Test CE | Paired accuracy change (pp) | Splits improving / 10 |
|---|---:|---:|---:|---:|---:|
| official | 0 | 49.68 ± 11.08 | 3.302 | — | — |
| official | 1 | 47.84 ± 11.12 | 3.792 | -1.84 | 4 |
| official | 2 | 47.19 ± 10.60 | 4.010 | -0.65 | 5 |
| official | 3 | 48.31 ± 10.45 | 3.275 | +1.12 | 8 |
| adversarial | 0 | 49.68 ± 11.08 | 3.302 | — | — |
| adversarial | 1 | 49.00 ± 10.45 | 2.800 | -0.68 | 6 |
| adversarial | 2 | 43.10 ± 9.18 | 5.347 | -5.90 | 2 |
| adversarial | 3 | 45.62 ± 11.25 | 5.214 | +2.52 | 6 |

Neither mean accuracy trajectory is monotonic in either direction: both decline at the first two additions and recover at the third. Neither test-CE trajectory is monotonic either. Official final accuracy is 1.37 pp below baseline; adversarial final accuracy is 4.06 pp below baseline. Official final CE is slightly below baseline despite lower accuracy.

These observations support “more graphs do not reliably help in this tested configuration,” not “more graphs always hurt.” Different source sets define the two paths; all rungs were trained independently, not continued from the preceding rung. This is therefore not an isolated test of ordering or history dependence for an identical source set. One pretraining seed and a baseline repeatability discrepancy do not establish robust path sensitivity.

## Protocol and deviations

- Target Computers, node classification, 1-shot: ten labeled support nodes, 1,374 validation nodes and 12,368 test nodes per split. Fixed cached [10,10] two-hop target neighborhoods and matching support/test identities across rungs.
- Shared five-source baseline: ogbn-arxiv, Reddit, FB15k_237, PROTEINS, HIV. Official additions: PubMed → Photo → FacebookPagePage. Adversarial additions: Photo → Lipophilicity → MUTAG. Rungs are cumulative sets.
- Training seed 0 only. One full released-trainer epoch per rung; native weighted source sampling and on-access source feature transforms. Mixed batch 512, 1,000 refinement rounds per node source, 200 local paths, 2,000 global paths × five rounds. No capped mixed batches in final runs.
- Input 128, hidden/attention 512, 32 generators, configured two GCN layers; released CLI pretraining lr 0.0003, k=5, dropout 0.1. Paper prose specifies lr 0.0001 and k=15. Exact Figure 5 configs were unavailable. Native model/indexing behavior is retained rather than the earlier pilot's geometry-index correction.
- One epoch is the native warm-up phase: prototype statistics update, but prototype-loss optimization begins only in epoch 2 and is absent here. This is a material limitation. Additional graphs also add training updates; this is not an update-matched comparison.
- Baseline reuses epoch 1 of the subsequently stopped 10-epoch run. First-epoch learning rate and model updates match; the scheduler state saved after epoch 1 differs from a one-epoch schedule. Only the pretrained model is loaded for adaptation.
- Adaptation: 2,000 epochs per split, batch 256, lr 0.001, dropout 0.2, alignment k=3, coefficient 1; validation every ten epochs, improvement threshold 0.001, no early stopping. Common adapter modules use matched initialization. Save validation-selected and terminal adapters; reported tests use selected adapters. Test accuracy and per-example CE come from one saved prediction pass. Test scores did not choose the adapter checkpoint.
- The initial split-0 test was inspected before the user chose the one-epoch ladder budget. Consequently the budget was not pre-registered independently of all observed target test performance.

## Reproduction gap and unresolved audit

The paper reports Computers 1-shot 59.5 ± 7.0 over ten runs with different random splits ([Table 1 and Appendix F.3](https://arxiv.org/html/2603.00618v1)). Our ten-split baseline is 49.68 ± 11.08, 9.82 pp below its mean. The earlier 57.99% test was one split, not a reproduction of the aggregate result. The later baseline rerun scores 59.68% on that same split (a 1.69 pp difference). Its cause remains unaudited; it must not be dismissed as established GPU noise. The previously observed single-prediction replay differences were much smaller. Thus small rung differences should not be claimed as robust effects.

No epoch-2 checkpoint was saved before the optional 10-epoch run was stopped. Training/validation loss convergence and repeatability need further scrutiny before stronger claims. No additional runs were launched for this write-up.

## Measured costs

Pretraining runtime includes native data loading within training; baseline time is recovered from its original first-epoch update trace, excluding checkpoint writing. Adaptation seconds sum the ten trial timers and include validation/test. These are stage timings, not total allocated GPU-hours or full campaign wall time. GPU memory is peak PyTorch allocated memory, not nvidia-smi usage. Reserved memory and full trial/config details are in the evidence JSON.

| Model | Pretraining min | Adaptation min | Peak allocated GiB, pretrain | Pretrain updates | Adapted parameters |
|---|---:|---:|---:|---:|---:|
| base | 27.43 | 22.20 | 16.00 | 4878 | 3,080,847 |
| official_1 | 31.85 | 22.18 | 15.97 | 5956 | 3,081,360 |
| official_2 | 37.60 | 21.83 | 15.99 | 6986 | 3,081,873 |
| official_3 | 42.89 | 21.73 | 15.98 | 8072 | 3,082,386 |
| adversarial_1 | 32.72 | 21.98 | 15.98 | 5908 | 3,081,360 |
| adversarial_2 | 32.58 | 22.27 | 15.97 | 5933 | 3,081,873 |
| adversarial_3 | 32.21 | 22.38 | 15.98 | 5936 | 3,082,386 |

Pretraining parameters: 2,449,408 for each model. Every model has 20,000 total adaptation updates across ten splits. Adaptation peak allocated memory is approximately 1.45 GiB. No smoke results appear in these tables.

## Evidence and exact artifact locations

Local evidence: [raw metrics and resolved configurations](data/evidence.json), [summary table](data/summary.csv), [validation receipt](data/validation_receipt.json). The update verified all 70 rows, recomputed aggregate accuracy, and matched recorded support IDs, test IDs/labels, and common-initialization hashes across rungs. This does not resolve the baseline rerun discrepancy or constitute a fresh recomputation from prediction tensors.

Producing worktree on Tucker: `/dataMeR1/phil/gfm/GraphGlue-ladder-oneepoch`, branch `codex/graphglue-ladder-oneepoch`, revision `414a7b1` (full revision in each config). Official upstream revision: `4f99ccb9eacaeff87f474aeeef364e5738898ffb`. Local producing worktree: `/Users/philipp/projects/gfm/GraphGlue-ladder-oneepoch`. Protocol/runner: `ladder/README.md`, `ladder/run.py`, `ladder/queue.sh` there.

Tucker-only result root: `/dataMeR1/phil/gfm/graphglue-runtime/ladder-oneepoch/seed_0/`. Substitute `<arm>` with `base`, `official_1`, `official_2`, `official_3`, `adversarial_1`, `adversarial_2`, or `adversarial_3`:

- Configs: `<root>/<arm>/pretrain_config.json`, `adapt_config.json`.
- Logs: `<root>/<arm>/run.log`, `pretrain_updates.jsonl` (except reused baseline), `wandb.log`.
- Pretraining checkpoints: `<root>/<arm>/checkpoints/pretrain_epoch_1.pth` and `pretrain_final_model.pth`, except baseline uses `/dataMeR1/phil/gfm/graphglue-runtime/full-baseline/seed_0/checkpoints/pretrain_epoch_1.pth`.
- Adaptation checkpoints: `<root>/<arm>/adapter_best_<split>.pth`, `adapter_terminal_<split>.pth`, split 0–9.
- Results: `<root>/<arm>/result.json`, `pretrain_metrics.json`, `trial_<split>.json`, `paired_test_<split>.pt`.
- Earlier diagnostic: `/dataMeR1/phil/gfm/graphglue-runtime/full-baseline/epoch1-validation/test_result.json`.
- [W&B project](https://wandb.ai/eibl-usc/graphglue); ladder run IDs follow `graphglue-ladder-oneepoch-seed0-<arm>-20260908`.
