# Final-core downstream classification ladder

`classification_long.tsv` contains the validated 2,500-step downstream
classification evaluation for the final-core PRODIGY ladder checkpoints.

- training seeds: 0, 1, 2
- mixture orders: A, B, C
- mixture sizes: 1 through 9
- held-out targets: COVID political, Election 2020, Facebook pages, TwiBot-20,
  and UKR–RUS suspended
- evaluation size: 128 fixed episodes per model–target cell
- total cells: 375 distinct physical checkpoint–target evaluations

The source run was produced on Tucker under
`/dataMeR1/phil/gfm/prodigy-cls2500/log/finalcore_cls2500_ladders_run2/` by
`scripts/experiments/setup/finalcore_cls2500_ladders/run_tucker.sh`. The
aggregation step verifies complete coverage, metric bounds, checkpoint step,
episode count, and identical episode fingerprints within each target.

`../../plot_nm_cls_auc_ladders.py` combines these classification AUC values
with the archived NM AUC values in
`../prodigy_final_core/log_recovered_metrics/physical_metrics.tsv`.
