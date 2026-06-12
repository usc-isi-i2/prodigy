#!/bin/bash
#SBATCH --job-name=ukr_rus_csv_to_parquet
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail

SRC_ROOT=/project2/ll_774_951/uk_ru/twitter/data
OUT_ROOT=/scratch1/eibl/data/ukr_rus_twitter/parquet

manifest="${MANIFEST:?pass MANIFEST=/path/to/files.txt via sbatch --export}"
in=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$manifest")

if [ -z "${in:-}" ]; then
  echo "No input for task $SLURM_ARRAY_TASK_ID"
  exit 0
fi

rel="${in#$SRC_ROOT/}"
out="$OUT_ROOT/${rel%.csv}.parquet"
tmp="${out}.tmp"

if [ -s "$out" ]; then
  echo "Already exists: $out"
  exit 0
fi

mkdir -p "$(dirname "$out")"

echo "Converting: $in"
echo "Output:     $out"

tr -d '\r' < "$in" | duckdb -c "
SET threads TO 2;

COPY (
  SELECT *
  FROM read_csv(
    '/dev/stdin',
    header = true,
    all_varchar = true,
    ignore_errors = false
  )
) TO '$tmp' (
  FORMAT PARQUET,
  COMPRESSION ZSTD
);
"

mv "$tmp" "$out"
