chmod +x generate_manifest.sh

./generate_manifest.sh 50 /scratch1/eibl/data/covid19_twitter/manifests/files.test50.txt

manifest=/scratch1/eibl/data/covid19_twitter/manifests/files.test50.txt
n=$(wc -l < "$manifest")

sbatch --export=ALL,MANIFEST="$manifest" --array=0-$((n - 1)) csv_to_parquet.sbatch