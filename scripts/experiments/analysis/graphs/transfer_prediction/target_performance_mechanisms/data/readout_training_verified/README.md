# Completed readout-training intervention and control audit

Source runtime, Tucker only:
`/dataMeR1/phil/gfm/prodigy-mechanisms-freeze/log/target_mechanisms/readout_constraint_training_20260906`.
Frozen training/evaluation revision `f20e6495576f42dfd24aae50b289b3b7bff8c394`.
Launched 2026-09-06 11:04:09 UTC; evaluation completed at 14:25:59 UTC after
all training gates. Eighteen CPU-only models: Ukraine/Hong Kong/COVID × three
seeds × free/frozen initial readout, 2500 updates and 10000 episodes each.

`DONE.json` is the training verifier's original completion receipt;
`evaluation_DONE.json` is the evaluation pipeline's completion receipt.
`arms.json`, `paired_input_hashes.json`, `input_validation.json`, `model_list.tsv`
and `pipeline.json` retain provenance and complete input checks. Ten raw replay
files are in the sibling `readout_training_original/` and `readout_training_fresh/`
directories. Checkpoints and full consumed-training logs remain on Tucker.

The prospective analyzer requires all 3060 rows, 180 full-model cells, nine
matched free/frozen pairs, three initializations and both streams. No test-query
fitting or target checkpoint selection. Run from the mechanism worktree:

```bash
/opt/homebrew/bin/python3.11 -m scripts.experiments.analysis.graphs.transfer_prediction.target_performance_mechanisms.analyze_readout_training
MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/gfm-mechanisms-mpl /opt/homebrew/bin/python3.11 -m scripts.experiments.analysis.graphs.transfer_prediction.target_performance_mechanisms.plot_readout_training
```

The full-model primary FAILED; the Facebook readout probe improves in all nine
pairs on both streams. All targets, stages, sources and seed-level reversals are
retained, including the favorable secondary COVID/Facebook result.

`control_reproduction.json` is the original six-control weight comparison.
`control_state_audit.json` adds read-only comparisons of all six controls at
updates 0/100/2500 and every field in all 2500 consumed episode-audit rows.
Saved initial state is identical, but parameters have diverged by update 100;
the largest terminal tensor differences are normalization variances. Matching
parent RNG, member IDs and context counts does not recover missing historical
full input hashes or worker RNG. Cross-launch drift remains unexplained.
These controls are not exact replicas or additional independent seeds.

The read-only reproducer, after moving source through private git, is:

```bash
CUDA_VISIBLE_DEVICES='' /home/mhchu/miniconda3/envs/prodigy/bin/python -m scripts.experiments.setup.target_performance_mechanisms.audit_readout_control_states --previous-run /dataMeR1/phil/gfm/prodigy-mechanisms-train/log/target_mechanisms/member_cpu_training_20260906 --current-run /dataMeR1/phil/gfm/prodigy-mechanisms-freeze/log/target_mechanisms/readout_constraint_training_20260906
```

It prints the audit to stdout without changing either run. Both the original
outcomes and the negative reproduction check must accompany interpretation.
