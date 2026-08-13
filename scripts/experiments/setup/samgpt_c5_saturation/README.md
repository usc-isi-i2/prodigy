# SAMGPT five-source convergence trajectory — setup

This record describes the completed 4,000-update convergence trajectory for the five-source
Order-C SAMGPT endpoint, evaluated on held-out TwiBot-20 bot classification.

The implementation lives in the private sibling repository `../samgpt-social` on branch
`codex/samgpt-c5-4k-trajectory`, commit `cf54151`. The source config is
`configs/mixture_order_c/r5_4k_trajectory.json`; the launcher is
`scripts/run_c5_4k_trajectory_tucker.sh`.

Run from the dedicated Tucker worktree:

```bash
cd /dataMeR1/phil/gfm/samgpt-c5-4k-trajectory
GPU_ID=0 bash scripts/run_c5_4k_trajectory_tucker.sh
```

Canonical output:

`/dataMeR1/phil/gfm/samgpt-c5-4k-trajectory/log/mixture_order_c_4k/samgpt_mix_order_c_r5_4k_trajectory_20260806T180301Z/`

Checkpoints and target embeddings remain on Tucker. Compact metrics and provenance are
committed under `analysis/transfer/ablations/saturation/samgpt/five_source/samgpt_c5_saturation/data/`.
