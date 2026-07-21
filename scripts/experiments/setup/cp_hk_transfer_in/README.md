# CP-HK Transfer-In Evaluation

Evaluate NM-trained source models from the recent transfer studies on the new
CP-HK retweet graph.

This is the mirror of `scripts/experiments/cp_hk_twitter`: CP-HK -> other
graphs has already been run; this folder runs other NM sources -> CP-HK.

## Scope

Target graph:

```text
/dataMeR1/phil/data/cp_hk_twitter/graphs/retweet_graph.pt
```

Target task:

- `neighbor_matching`
- 3-shot
- 3-way and 30-way

Source checkpoints are matched-compute checkpoints only. For merged models, this
means the 50k checkpoint rather than the full 110k checkpoint.

Included sources:

- `nm_matrix_ukr`
- `nm_matrix_covid`
- `nm_matrix_merged_match`
- `nm_xsrc_within_source_match`
- `nm_cm_midterm`
- `nm_cm_covid`
- `nm_cm_merged_match`
- `nm_cm_within_match`
- `nm_cm_within_balanced_match`
- `nm_twibot20`

## Run On Tucker

Use only physical GPUs `0,1,2,3`.

```bash
cd /dataMeR1/phil/gfm/prodigy
GPUS=1,3 bash scripts/experiments/cp_hk_transfer_in/eval_sources_on_cp_hk_tucker.sh
```

For a dry run:

```bash
DRY_RUN=1 GPUS=1,3 bash scripts/experiments/cp_hk_transfer_in/eval_sources_on_cp_hk_tucker.sh
```

Results land under:

```text
/dataMeR1/phil/gfm/prodigy/log/eval_*_to_cp_hk_twitter_nm_3shot_3way_*
/dataMeR1/phil/gfm/prodigy/log/eval_*_to_cp_hk_twitter_nm_3shot_30way_*
```

