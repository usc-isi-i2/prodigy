./generate_manifest.sh all /scratch1/eibl/data/covid19_twitter/manifests/files.all.txt

manifest=/scratch1/eibl/data/covid19_twitter/manifests/files.all.txt
n=$(wc -l < "$manifest")

mkdir -p logs

sbatch \
  --export=ALL,MANIFEST="$manifest" \
  --array=0-$((n - 1))%20 \
  csv_to_parquet.sbatch