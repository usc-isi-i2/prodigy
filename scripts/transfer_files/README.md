# Transfer Files

Small helper scripts for transferring large dataset trees onto Tucker.

## Ukraine-Russia Twitter

Transfers the staged parquet tree from CARC storage to Tucker:

- source: `/scratch1/eibl/data/ukr_rus_twitter/parquet/`
- destination: `/dataMeR2/phil/data/ukr_rus_twitter/parquet/`

Run:

```bash
bash scripts/transfer_files/transfer_ukr_rus_parquet_to_tucker.sh
```

## COVID-19 Twitter

Transfers the staged parquet tree from CARC storage to Tucker:

- source: `/scratch1/eibl/data/covid19_twitter/parquet/`
- destination: `/dataMeR2/phil/data/covid19_twitter/parquet/`

Run:

```bash
bash scripts/transfer_files/transfer_covid19_parquet_to_tucker.sh
```

## Selected Graph Batch Transfer

Transfers only the `graphs` directories for the following datasets from CARC storage to Tucker:

- `/scratch1/eibl/data/covid19_twitter/graphs/`
- `/scratch1/eibl/data/covid_political/graphs/`
- `/scratch1/eibl/data/election2020/graphs/`
- `/scratch1/eibl/data/midterm/graphs/`
- `/scratch1/eibl/data/ukr_rus_suspended/graphs/`
- `/scratch1/eibl/data/ukr_rus_twitter/graphs/`

Each dataset is copied into the matching destination under `/dataMeR2/eibl/data/<dataset>/graphs/`.

Run:

```bash
bash scripts/transfer_files/transfer_selected_datasets_to_tucker.sh
```

## Notes

- All scripts use `hpc-transfer1.usc.edu` and `rsync --partial --info=progress2`.
- The trailing `/` on the source path copies the source directory contents into the destination directory.
- Update the username or paths if the cluster layout changes.
