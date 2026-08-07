# SAMGPT sampled-COVID saturation trajectory — setup

This is the completed 0–4,000-update trajectory on the deterministic 150,000-node
`covid19_twitter` runtime view used by the SAMGPT nine-source ladder. TwiBot-20 is held out
and evaluated with fixed 10-shot validation episodes.

Implementation: private sibling repository `../samgpt-social`, branch
`codex/samgpt-covid-saturation-4k`, commit `a9c5e5d`. Config:
`configs/pretrain_saturation/covid_4k_trajectory.json`. Launcher:
`scripts/run_covid_saturation_4k_tucker.sh`.

```bash
cd /dataMeR1/phil/gfm/samgpt-covid-saturation-4k
GPU_ID=1 bash scripts/run_covid_saturation_4k_tucker.sh full
```

Canonical output:

`/dataMeR1/phil/gfm/samgpt-covid-saturation-4k/log/pretrain_saturation_covid_4k/samgpt_covid19_twitter_4k_trajectory_20260806T190046Z/`

This is sampled-graph SAMGPT evidence, not a full-graph reproduction of PRODIGY's COVID
arm. Compact evidence is committed in the paired analysis folder; weights remain on Tucker.
