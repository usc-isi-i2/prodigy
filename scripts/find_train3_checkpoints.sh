#!/bin/bash
# Run on cluster: bash scripts/find_train3_checkpoints.sh > scripts/model_lists/train3_all_models.txt
# Errors (missing checkpoints) go to stderr.
set -euo pipefail
cd /home1/eibl/gfm/prodigy

echo "# Train3 models — latest checkpoints (auto-generated)"

while IFS='|' read -r name prefix; do
  latest=$(ls "state/${prefix}/checkpoint/state_dict_"*.ckpt 2>/dev/null | sort -V | tail -1)
  if [[ -n "$latest" ]]; then
    printf "%-30s  %s\n" "$name" "$latest"
  else
    echo "# MISSING: $prefix" >&2
  fi
done << 'MODELS'
exp1_mid_cov_ukr|train3_exp1_midterm_nm_to_covid_nm_to_ukr_rus_nm_28_04_2026_07_05_55
exp1_mid_cov_elec|train3_exp1_midterm_nm_to_covid_nm_to_election2020_nm_28_04_2026_07_05_55
exp1_mid_cov_ukrs|train3_exp1_midterm_nm_to_covid_nm_to_ukr_rus_suspended_nm_28_04_2026_07_05_55
exp1_mid_cov_cpol|train3_exp1_midterm_nm_to_covid_nm_to_covid_political_nm_28_04_2026_07_05_55
exp2_mid_ukr_cov|train3_exp2_midterm_nm_to_ukr_rus_nm_to_covid_nm_28_04_2026_07_05_55
exp2_mid_ukr_ukrs|train3_exp2_midterm_nm_to_ukr_rus_nm_to_ukr_rus_suspended_nm_28_04_2026_07_05_55
exp2_mid_ukr_elec|train3_exp2_midterm_nm_to_ukr_rus_nm_to_election2020_nm_28_04_2026_07_05_55
exp2_mid_ukr_cpol|train3_exp2_midterm_nm_to_ukr_rus_nm_to_covid_political_nm_28_04_2026_07_05_55
exp3_cov_ukr_mid|train3_exp3_covid_nm_to_ukr_rus_nm_to_midterm_nm_28_04_2026_07_05_55
exp3_cov_ukr_elec|train3_exp3_covid_nm_to_ukr_rus_nm_to_election2020_nm_28_04_2026_07_06_26
exp3_cov_ukr_ukrs|train3_exp3_covid_nm_to_ukr_rus_nm_to_ukr_rus_suspended_nm_28_04_2026_07_05_56
exp3_cov_ukr_cpol|train3_exp3_covid_nm_to_ukr_rus_nm_to_covid_political_nm_28_04_2026_07_06_05
MODELS