# SAMGPT COVID correctness ablation — setup

This completed diagnostic separates three issues in the sampled-COVID SAMGPT trajectory:
the high zero-update baseline, the inherited structure-prompt routing bug, and reuse of fixed
GraphCL corruption views.

Implementation: private sibling repository `../samgpt-social`, branch
`codex/samgpt-covid-correctness-ablation`, commit `332a1d2`. The two corrected 500-update
configs and zero-step evaluator are under `configs/pretrain_saturation/` and
`samgpt_social/zero_step_controls.py` on that branch.

```bash
cd /dataMeR1/phil/gfm/samgpt-covid-correctness-ablation
GPU_ID=0 bash scripts/run_covid_correctness_ablation_tucker.sh controls
GPU_ID=0 bash scripts/run_covid_correctness_ablation_tucker.sh b full
GPU_ID=0 bash scripts/run_covid_correctness_ablation_tucker.sh c full
```

Canonical outputs are under:

`/dataMeR1/phil/gfm/samgpt-covid-correctness-ablation/log/covid_correctness_ablation/`

The original Arm A values come from the paired sampled-COVID saturation experiment. Arms B
and C, zero-step controls, and provenance are committed in the analysis `data/` folder.
