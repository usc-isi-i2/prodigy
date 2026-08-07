# SAMGPT weak-to-strong mixture ladder — setup

This completed five-rung experiment orders sources by their specialist transfer to held-out
TwiBot-20 and tests whether mixture performance follows the cumulative specialist maximum.

Implementation: private sibling repository `../samgpt-social`, branch
`codex/samgpt-weak-to-strong`, experiment commit `cca7064`; archival analysis commit
`9a5f02b`. Configs are in `configs/mixture_weak_to_strong/` and the Tucker launcher is
`scripts/run_mixture_weak_to_strong_tucker.sh`.

The canonical C2–C4 run directories live in the dedicated Tucker worktree
`/dataMeR1/phil/gfm/samgpt-social-w2s`; C1 and C5 reuse the registered specialist and
five-source endpoints. Checkpoints and episode-level outputs remain on Tucker and W&B.

The complete compact result package—summary tables, paired statistics, figure, and analysis
script—is copied into `analysis/samgpt_weak_to_strong/` in this repository.
