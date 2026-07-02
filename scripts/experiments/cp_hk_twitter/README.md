# CP-HK Twitter NM

This experiment trains neighbor matching on the COSINE 2022 CP-HK Twitter
retweet graph, using profile bios as node features.

Source data on CARC:

```text
/project2/emiliofe_74/data_backup/COSINE/2022CP-HK/an_cp-hk.twitter.v7-ground-truth.2020-04-07_2020-08-23.json.gz
/project2/emiliofe_74/data_backup/COSINE/2022CP-HK/an_cp-hk.twitter.v7-ground-truth.2020-08-24_2020-09-13.json.gz
```

Tucker target paths for this run use `/dataMeR1` by request:

```text
/dataMeR1/phil/data/cp_hk_twitter/raw
/dataMeR1/phil/data/cp_hk_twitter/parquet
/dataMeR1/phil/data/cp_hk_twitter/embeddings
/dataMeR1/phil/data/cp_hk_twitter/graphs
/dataMeR1/phil/logs
```

The build pipeline is:

1. Copy CP-HK raw `.json.gz` files from CARC to Tucker.
2. Extract retweet events and latest non-empty user bios to parquet with
   `scripts/graph_construction/cp_hk_json_to_parquet.py`.
3. Embed `user_bios.parquet:profile` with `tweet-embeddings-v001`.
4. Build `retweet_graph.pt` with `x = bio_emb_*`, weighted retweet edges, and
   temporal history/future edge views.
5. Train NM with `prodigy`.

Tucker does not use Slurm for this workflow. Use `run_cp_hk_nm_tucker.sh`, which
launches build/train phases directly and can be wrapped in `nohup`.

GPU use is restricted to physical GPUs `0,1,2,3`. Set `GPU_ID` to one of those
values; the runner masks to that single GPU and the training config uses
`device: 0` inside the masked process.

For the first overnight run, it is acceptable to build from only the smaller
CP-HK shard while the larger shard is still expensive to transfer:

```bash
CP_HK_RAW_FILES="an_cp-hk.twitter.v7-ground-truth.2020-08-24_2020-09-13.json.gz" \
GPU_ID=1 bash scripts/experiments/cp_hk_twitter/run_cp_hk_nm_tucker.sh
```

From the local machine, transfer raw files with:

```bash
bash scripts/experiments/cp_hk_twitter/transfer_cp_hk_raw_to_tucker.sh
```

## Transfer Evaluation

Evaluate the CP-HK NM checkpoint on the requested transfer grid with:

```bash
GPUS=1 bash scripts/experiments/cp_hk_twitter/eval_cp_hk_transfer_tucker.sh
```

The runner performs:

- NM, 3-shot, 3-way on `covid19_twitter`, `ukr_rus_twitter`, `midterm`,
  `cp_hk_twitter`, `twibot20`, `covid_political`, `election2020`,
  `ukr_rus_suspended`.
- NM, 3-shot, 30-way on the same graph list.
- NC, 3-shot on `twibot20`, `covid_political`, `election2020`,
  `ukr_rus_suspended`.

Use `DRY_RUN=1` to print the commands. `GPUS` must be a comma-separated subset
of physical GPUs `0,1,2,3`.

### 2026-07-02 CP-HK Transfer Results

Model: `/dataMeR1/phil/gfm/prodigy/state/cp_hk_twitter_nm_bio_02_07_2026_08_58_57/state_dict`

Queue log: `/dataMeR1/phil/logs/cp_hk_transfer_eval_20260702_101456.log`

CP-HK is not primarily COVID-related. A raw-shard term sample is dominated by
Hong Kong/protest terms (`HongKong`, `FreeHongKong`, police, protest, CCP,
China); COVID terms appear much less often and are likely incidental to the 2020
collection window.

| Task | Dataset | n-way | Test acc | Test F1 | Test ROC-AUC |
|------|---------|------:|---------:|--------:|-------------:|
| NM | covid19_twitter | 3 | 0.6334 | 0.6333 | 0.8226 |
| NM | ukr_rus_twitter | 3 | 0.7112 | 0.7111 | 0.8798 |
| NM | midterm | 3 | 0.5845 | 0.5844 | 0.8131 |
| NM | cp_hk_twitter | 3 | 0.6663 | 0.6663 | 0.8627 |
| NM | twibot20 | 3 | 0.6399 | 0.6398 | 0.8238 |
| NM | covid_political | 3 | 0.5112 | 0.5110 | 0.7165 |
| NM | election2020 | 3 | 0.4920 | 0.4918 | 0.6731 |
| NM | ukr_rus_suspended | 3 | 0.5087 | 0.5083 | 0.7085 |
| NM | covid19_twitter | 30 | 0.1416 | 0.1399 | 0.7530 |
| NM | ukr_rus_twitter | 30 | 0.1702 | 0.1675 | 0.8045 |
| NM | midterm | 30 | 0.1382 | 0.1384 | 0.7310 |
| NM | cp_hk_twitter | 30 | 0.1382 | 0.1343 | 0.7910 |
| NM | twibot20 | 30 | 0.1240 | 0.1207 | 0.7627 |
| NM | covid_political | 30 | 0.0788 | 0.0757 | 0.6675 |
| NM | election2020 | 30 | 0.0702 | 0.0692 | 0.6323 |
| NM | ukr_rus_suspended | 30 | 0.0833 | 0.0809 | 0.6481 |
| NC | twibot20 | 2 | 0.5725 | 0.5964 | 0.6090 |
| NC | covid_political | 2 | 0.5803 | 0.6057 | 0.6054 |
| NC | election2020 | 2 | 0.5380 | 0.5157 | 0.5308 |
| NC | ukr_rus_suspended | 2 | 0.6520 | 0.6258 | 0.6973 |
