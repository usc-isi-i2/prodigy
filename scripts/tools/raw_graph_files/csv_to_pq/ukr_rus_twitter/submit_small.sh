#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$script_dir/generate_manifest.sh" 50 /scratch1/eibl/data/ukr_rus_twitter/manifests/files.test50.txt

manifest=/scratch1/eibl/data/ukr_rus_twitter/manifests/files.test50.txt
n=$(wc -l < "$manifest")

mkdir -p "$script_dir/logs"

sbatch \
  --chdir="$script_dir" \
  --export=ALL,MANIFEST="$manifest" \
  --array=0-$((n - 1)) \
  "$script_dir/csv_to_parquet.sbatch"
