#!/usr/bin/env bash
# Read-only check that all canonical rung artifacts already exist on Tucker.
set -euo pipefail

INPUTS=(
  /dataMeR1/phil/data/ukr_rus_twitter/graphs/retweet_graph_parquet.pt
  /dataMeR1/phil/data/merged/graphs/ukr_rus_covid_retweet_graph.pt
  /dataMeR1/phil/data/merged/graphs/ukr_rus_covid_midterm_retweet_graph.pt
  /dataMeR1/phil/data/merged/graphs/ukr_rus_covid_midterm_4src_retweet_graph.pt
  /dataMeR1/phil/data/merged/graphs/ukr_rus_covid_midterm_5src_retweet_graph.pt
  /dataMeR1/phil/data/merged/graphs/ukr_rus_covid_midterm_6src_retweet_graph.pt
  /dataMeR1/phil/data/merged/graphs/ukr_rus_covid_midterm_7src_retweet_graph.pt
  /dataMeR1/phil/data/merged/graphs/ukr_rus_covid_midterm_all8_retweet_graph.pt
)

missing=0
for path in "${INPUTS[@]}"; do
  if [[ -r "${path}" ]]; then
    stat --printf='OK %s bytes %n\n' "${path}"
  else
    echo "MISSING ${path}" >&2
    missing=1
  fi
done
exit "${missing}"
