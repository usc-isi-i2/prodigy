#!/bin/bash
# Quick reference for cross-dataset evaluation on cluster

# ============================================================================
# STEP 1: Generate Jobs (do once)
# ============================================================================

cd /home1/eibl/gfm/prodigy
python scripts/generate_eval_jobs.py

# Creates:
# - eval_jobs.txt          (list of 90 evaluation commands)
# - eval_cross_dataset.sbatch  (SLURM submission script)

# ============================================================================
# STEP 2: Submit to Cluster
# ============================================================================

# Review jobs
cat eval_jobs.txt | head -20

# Submit
sbatch eval_cross_dataset.sbatch

# Check submission
squeue -u $USER
echo "Watch this counter go up:"
watch -n 5 "find /home1/eibl/gfm/prodigy/eval_results -name '*.json' | wc -l"

# ============================================================================
# STEP 3: Monitor Progress
# ============================================================================

# Stream latest logs
tail -f /home1/eibl/gfm/prodigy/logs/eval_*.log

# Count completed jobs
find /home1/eibl/gfm/prodigy/eval_results -name "*.json" | wc -l

# Check job status
squeue -u $USER -O "JOBID,ARRAY_TASK_ID,STATE,TIME,NODELIST"

# ============================================================================
# STEP 4: Aggregate Results (after jobs finish)
# ============================================================================

python scripts/aggregate_eval_results.py \
  --results_dir /home1/eibl/gfm/prodigy/eval_results

# Generates:
# - summary.csv            (all results in table format)
# - summary.json           (structured results with rankings)
# - Console output         (high-level stats)

# ============================================================================
# OPTIONAL: Run Single Evaluation (for testing)
# ============================================================================

python scripts/eval_cross_dataset.py \
  --model_path /scratch1/singhama/data/experiments/train2_midterm_nm_to_covid_nm_16_04_2026_10_07_00/state_dict \
  --dataset covid19_twitter \
  --task node_masking \
  --device 0 \
  --output_dir /tmp/test_eval

# ============================================================================
# TROUBLESHOOTING
# ============================================================================

# Cancel all jobs
scancel -n cross_eval

# Check GPU availability
nvidia-smi

# View job error log
cat /home1/eibl/gfm/prodigy/logs/eval_0.err

# Debug: count evaluations by dataset
for ds in midterm covid19_twitter ukr_rus_twitter; do
  echo -n "$ds: "
  find /home1/eibl/gfm/prodigy/eval_results/$ds -name "*.json" 2>/dev/null | wc -l
done

# ============================================================================
# EXPECTED RESULTS
# ============================================================================

# 15 models × 3 datasets × 2 tasks = 90 evaluations
# Runtime: ~2-4 hours total with 40 parallel jobs

# Results location:
# /home1/eibl/gfm/prodigy/eval_results/{dataset}/{task}/eval_*.json

# Example result file:
# /home1/eibl/gfm/prodigy/eval_results/midterm/node_masking/eval_midterm_node_masking_26_04_2026_15_30_45.json

# ============================================================================
