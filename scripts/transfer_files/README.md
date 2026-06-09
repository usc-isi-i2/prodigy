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

## Selected Dataset Batch Transfer

Transfers the following dataset trees from CARC storage to Tucker:

- `/scratch1/eibl/data/covid19_twitter/`
- `/scratch1/eibl/data/ed/`
- `/scratch1/eibl/data/ukr_rus_suspended/`
- `/scratch1/eibl/data/covid_masking/`
- `/scratch1/eibl/data/election2020/`
- `/scratch1/eibl/data/covid_political/`
- `/scratch1/eibl/data/immigration_julia/`
- `/scratch1/eibl/data/social_llm_covid/`

Each dataset is copied into the matching destination under `/dataMeR2/phil/data/<dataset>/`.

Run:

```bash
bash scripts/transfer_files/transfer_selected_datasets_to_tucker.sh
```

## Social LLM Data

Transfers the `social_llm_data` directory from CARC project storage to Tucker:

- source: `/project2/emiliofe_74/julie/social_llm_data`
- destination: `/dataMeR2/phil/data/social_llm_data`

Run:

```bash
bash scripts/transfer_files/transfer_social_llm_data_to_tucker.sh
```

## Notes

- All scripts use `hpc-transfer1.usc.edu` and `rsync --partial --info=progress2`.
- The trailing `/` on the source path copies the source directory contents into the destination directory.
- Update the username or paths if the cluster layout changes.
