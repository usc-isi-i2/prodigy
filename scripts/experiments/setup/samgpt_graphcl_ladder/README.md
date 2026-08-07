# SAMGPT nine-source native-GraphCL ladder — setup

This completed experiment evaluates the nine-source, three-order SAMGPT ladder and the
nine specialists on SAMGPT's native GraphCL discrimination objective: 27 ladder models × 9
targets and 9 specialists × 9 targets.

Implementation: private sibling repository `../samgpt-social`, branch
`codex/samgpt-native-objective-eval`, commit `b8bc122`. The ladder and specialist configs are
under `configs/nm_ladder_9x3/` and `configs/single_source_nm_matrix/`; the CARC evaluation
launchers are in `scripts/slurm_graphcl_ladder_eval_carc.sh` and
`scripts/run_all_graphcl_single_source_matrix_tucker.sh`.

The compact exported matrices and rule-comparison tables are committed in the paired
analysis folder. Raw checkpoints and full runtime logs remain on CARC/Tucker as recorded in
the manifests.
