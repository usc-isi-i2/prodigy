# Fair-two-hop ladder downstream analysis

Analysis for `setup/nm_ladder_downstream_nhop2/`. Add `FINDINGS.md` only after the complete
39-encoder sweep has passed both task gates.

Registered primary metrics are classification ROC-AUC and repaired static-LP AUC with
degree-matched negatives. The assembler produces:

- `data/downstream_long.csv`: all metrics over 40 logical ladder rows;
- `data/classification_roc_auc.csv` and `data/static_lp_auc.csv`: primary wide tables;
- `data/entry_jumps.csv`: each eligible graph immediately before/after entering;
- `data/paired_to_matched40k.csv`: sequential, split-aware, and fixed-exposure Order A
  paired cellwise with matched-40k Order A;
- `data/pair_lp_floors.csv`: heuristic/raw-feature floors on the shared pair sets; and
- `data/summary.json`: descriptive entry summaries and completeness metadata.

The plotting script produces three PNG/PDF pairs in `figures/`:

- `entry_jumps`: registered before/after entry effects across all trajectories;
- `rung_trajectories`: every primary-metric trajectory with graph-entry markers; and
- `controlled_vs_matched40k`: paired schedule/split/exposure deltas by graph role.

Expected complete counts are 40 entry events (19 classification, 21 static LP) and 216
controlled variant-vs-matched40k cells. These are paired measurements from one training
seed, not independent replicates.

Reassemble on Tucker:

```bash
python3 scripts/experiments/analysis/transfer/ablations/downstream/two_hop/nm_ladder_downstream_nhop2/assemble_results.py
python3 scripts/experiments/analysis/transfer/ablations/downstream/two_hop/nm_ladder_downstream_nhop2/plot_results.py
```
