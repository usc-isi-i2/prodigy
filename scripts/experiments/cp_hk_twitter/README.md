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

From the local machine, transfer raw files with:

```bash
bash scripts/experiments/cp_hk_twitter/transfer_cp_hk_raw_to_tucker.sh
```
