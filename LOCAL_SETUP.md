# Local Setup Snapshot

Date: 2026-02-06
Git branch: main
Git base commit: 375d9d5

## Environment (prodigy_clean)

```bash
cd /Users/philipp/projects/gfm/prodigy
source prodigy_clean/bin/activate
```

## Dependency Notes

- Python: 3.10.x (recommended for this repo/toolchain)
- Torch stack used during sanity checks:
  - `torch==2.0.1`
  - `torch-geometric==2.3.1`
  - `torch-scatter==2.1.2`
  - `torch-sparse==0.6.18`
  - `torch-cluster==1.6.3`
- If `sentence-transformers==2.2.2` import fails with `cached_download`, pin:
  - `huggingface_hub==0.14.1`

## Sanity Run Command (default dataset)

```bash
WANDB_MODE=offline python experiments/run_single_experiment.py \
  --dataset arxiv \
  --root /Users/philipp/FSdatasets \
  --original_features True \
  --input_dim 128 \
  -ds_cap 10 -val_cap 5 -test_cap 5 \
  --epochs 1 -eval_step 5 -ckpt_step 5 \
  -task classification \
  -bs 1 -way 3 -shot 1 -qry 1 \
  --device 123 \
  --prefix SANITY_ARXIV
```

## Custom Twitter CSV Pretraining Command

```bash
WANDB_MODE=offline python experiments/run_single_experiment.py \
  --dataset twitter \
  --root /Users/philipp/Downloads/converted_data_2022-10-04-00_2022-10-04-23 \
  --csv_filename midterm-2022-10-04-06.csv \
  --label_type verified \
  --max_users 1000 \
  --original_features True \
  --input_dim 3 \
  -ds_cap 1000 -val_cap 100 -test_cap 100 \
  --epochs 1 -eval_step 100 -ckpt_step 100 \
  -task classification \
  -bs 1 -way 2 -shot 3 -qry 4 \
  --device 123 \
  --prefix TWITTER_PT
```

Notes:
- For twitter numeric features in this loader, `--input_dim 3` matches `[followers_count, friends_count, statuses_count]`.
- Keep `--device 123` on macOS CPU.
