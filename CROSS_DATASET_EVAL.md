# Cross-Dataset Transfer Evaluation

This guide explains how to evaluate trained models across multiple datasets and tasks on the cluster.

## Overview

The complete evaluation pipeline:

1. **Collect** all trained model paths
2. **Generate** evaluation jobs for all model × dataset × task combinations
3. **Submit** jobs to cluster via SLURM array job
4. **Monitor** job progress in real-time
5. **Aggregate** results and analyze cross-dataset performance

This is the standard way to test out-of-distribution generalization in this repo.

---

## Quick Start

### 1. Generate Evaluation Jobs

```bash
cd /home1/eibl/gfm/prodigy
python scripts/generate_eval_jobs.py
```

This creates:
- `eval_jobs.txt` - List of all evaluation commands (90 jobs total)
- `eval_cross_dataset.sbatch` - SLURM batch submission script

### 2. Submit to Cluster

```bash
# Review the job list first
cat eval_jobs.txt

# Submit array job
sbatch eval_cross_dataset.sbatch
```

### 3. Monitor Progress

```bash
# Check job status
squeue -u $USER

# Stream logs
tail -f /home1/eibl/gfm/prodigy/logs/eval_*.log

# Count completed results
find /home1/eibl/gfm/prodigy/eval_results -name "*.json" | wc -l
```

### 4. Aggregate Results

```bash
python scripts/aggregate_eval_results.py \
  --results_dir /home1/eibl/gfm/prodigy/eval_results
```

---

## Detailed Setup

### Your Models

You have 15 trained models to evaluate:

| # | Model | Source→Target | Task |
|---|-------|---------------|------|
| 1 | train2_midterm_nm_to_covid_nm | midterm→covid | node_masking |
| 2 | exp2_train2_midterm_nm_to_ukr_rus_nm | midterm→ukr_rus | node_masking |
| 3 | exp3_train2_covid_nm_to_ukr_rus_nm | covid→ukr_rus | node_masking |
| 4 | exp4_train2_midterm_lp_to_covid_lp | midterm→covid | link_prediction |
| 5 | exp5_train2_midterm_lp_to_ukr_rus_lp | midterm→ukr_rus | link_prediction |
| 6 | exp6_train2_covid_lp_to_ukr_rus_lp | covid→ukr_rus | link_prediction |
| 7 | exp7_train2_midterm_nm_to_covid_lp | midterm→covid | cross-task |
| 8 | exp8_train2_midterm_nm_to_ukr_rus_lp | midterm→ukr_rus | cross-task |
| 9 | exp9_train2_covid_nm_to_ukr_rus_lp | covid→ukr_rus | cross-task |
| 10 | exp10_train2_midterm_lp_to_covid_nm | midterm→covid | cross-task |
| 11 | exp11_train2_midterm_lp_to_ukr_rus_nm | midterm→ukr_rus | cross-task |
| 12 | exp12_train2_covid_lp_to_ukr_rus_nm | covid→ukr_rus | cross-task |
| 13 | exp13_train2_covid_nm_to_midterm_nm | covid→midterm | node_masking |
| 14 | exp14_train2_ukr_rus_nm_to_midterm_nm | ukr_rus→midterm | node_masking |
| 15 | exp15_train2_ukr_rus_nm_to_covid_nm | ukr_rus→covid | node_masking |

### Evaluation Configuration

By default, all 15 models are evaluated on:
- **Datasets**: midterm, covid19_twitter, ukr_rus_twitter
- **Tasks**: node_masking, link_prediction
- **Total jobs**: 15 × 3 × 2 = **90 evaluations**

To customize, edit `scripts/generate_eval_jobs.py`:

```python
MODELS = [
    # Add/remove model paths here
]

DATASETS = [
    # Change which datasets to evaluate on
    "midterm",
    "covid19_twitter",
    "ukr_rus_twitter",
]

TASKS = [
    # Change which tasks to evaluate on
    "node_masking",
    "link_prediction",
]
```

Then regenerate:
```bash
python scripts/generate_eval_jobs.py \
  --job_list eval_jobs.txt \
  --sbatch_script eval_cross_dataset.sbatch
```

---

## Cluster Configuration

### SLURM Settings

The generated `eval_cross_dataset.sbatch` is configured with:

```
--array=0-89%16          # 90 jobs, max 16 parallel
--nodes=1
--ntasks=1
--cpus-per-task=8
--gres=gpu:1             # 1 GPU per job
--time=4:00:00           # 4 hours per job
--mem=32G
```

### Customize for Your Cluster

```bash
# Request a specific GPU type if the generic request is rejected
python scripts/generate_eval_jobs.py --gpu_type p100

# Run fewer jobs in parallel
python scripts/generate_eval_jobs.py --array_parallel 8

# Increase memory
python scripts/generate_eval_jobs.py --mem 64G

# Extend time limit to 6 hours
python scripts/generate_eval_jobs.py --time_limit 6:00:00
```

Notes:
- Each array task runs with `--device 0`. SLURM usually exposes the allocated GPU as local device `0` through `CUDA_VISIBLE_DEVICES`.
- If your cluster rejects `--gres=gpu:1`, regenerate with a typed request such as `--gpu_type p100`, `--gpu_type a100`, or another GPU type available on your partition.

---

## Single Job Evaluation (Testing)

To test a single model/dataset/task combination:

```bash
python scripts/eval_cross_dataset.py \
  --model_path /home1/eibl/gfm/prodigy/log/train2_midterm_nm_to_covid_nm_16_04_2026_10_07_00/state_dict \
  --dataset covid19_twitter \
  --task node_masking \
  --output_dir /home1/eibl/gfm/prodigy/eval_results/covid19_twitter/node_masking \
  --device 0 \
  --batch_size 5 \
  --dataset_len_cap 10000
```

**Options:**
- `--model_path` (required): Path to checkpoint
- `--dataset` (required): Dataset name
- `--task` (required): Task name
- `--device`: GPU ID (default: 0)
- `--batch_size`: Evaluation batch size (default: 5)
- `--dataset_len_cap`: Number of samples to evaluate (default: 10000)
- `--output_dir`: Where to save results (default: ./eval_results)
- `--force_cache`: Use cached dataset (default: false)
- `--root`: Root data directory (default: /home1/eibl/gfm/prodigy/FSdatasets)

---

## Results

### Output Structure

```
/home1/eibl/gfm/prodigy/eval_results/
├── midterm/
│   ├── node_masking/
│   │   └── eval_midterm_node_masking_DD_MM_YYYY_HH_MM_SS.json
│   └── link_prediction/
│       └── eval_midterm_link_prediction_DD_MM_YYYY_HH_MM_SS.json
├── covid19_twitter/
│   ├── node_masking/
│   └── link_prediction/
├── ukr_rus_twitter/
│   ├── node_masking/
│   └── link_prediction/
├── summary.csv           # Aggregated table
└── summary.json          # Structured summary
```

### Result Files

Each evaluation generates a JSON file:

```json
{
  "model_path": "/home1/eibl/gfm/prodigy/log/train2_midterm_nm_to_covid_nm_16_04_2026_10_07_00/state_dict",
  "dataset": "covid19_twitter",
  "task": "node_masking",
  "timestamp": "2026-04-26T15:30:45.123456",
  "val_results": {
    "accuracy": 0.92,
    "f1_score": 0.89,
    "auc_roc": 0.94
  },
  "test_results": {
    "accuracy": 0.89,
    "f1_score": 0.86,
    "auc_roc": 0.91
  }
}
```

### Aggregate Summary

After all jobs complete, run:

```bash
python scripts/aggregate_eval_results.py \
  --results_dir /home1/eibl/gfm/prodigy/eval_results
```

This generates:
- **summary.csv**: Tabular format with all results
- **summary.json**: Structured JSON with model rankings
- **Console output**: High-level statistics

---

## Monitoring & Troubleshooting

### Monitor Jobs

```bash
# Check job status
squeue -u $USER

# Check specific job
squeue -j <job_id>

# Stream specific log
tail -f /home1/eibl/gfm/prodigy/logs/eval_*.out

# Count completed evaluations
find /home1/eibl/gfm/prodigy/eval_results -name "*.json" | wc -l

# Monitor all jobs
watch -n 5 "squeue -u \$USER"
```

### Common Issues

**Jobs stuck in PENDING**
```bash
# Check for resource constraints
sinfo
squeue -u $USER -O "NAME,CPUS,MIN_CPUS,MIN_TMP_DISK,MIN_MEMORY,NODES,PRIORITY,STATE"
```

**Out of memory errors**
```bash
# Check GPU usage
nvidia-smi

# Run with smaller batch size
python scripts/eval_cross_dataset.py --batch_size 2 --dataset_len_cap 5000
```

**Model checkpoint not found**
```bash
# Verify checkpoint exists
ls -la /scratch1/singhama/data/experiments/MODEL_NAME/state_dict
ls -la /scratch1/singhama/data/experiments/MODEL_NAME/checkpoint/*.ckpt
```

**Dataset loading errors**
```bash
# Check if dataset is cached
ls -la /scratch1/singhama/data/FSdatasets/

# Clear cache and try again
python scripts/eval_cross_dataset.py --force_cache false --dataset midterm ...
```

**Cancel running jobs**
```bash
# Cancel all array jobs
scancel -n cross_eval

# Cancel specific job
scancel <job_id>

# Cancel range of jobs
scancel 12345-12350
```

---

## Advanced Usage

### Resume Interrupted Evaluations

If some jobs failed, check which combinations were completed:

```bash
# List all completed evaluations
python -c "
import json, os
from pathlib import Path
completed = set()
for f in Path('/scratch1/singhama/data/eval_results').rglob('*.json'):
    try:
        data = json.load(open(f))
        key = (data['model_path'], data['dataset'], data['task'])
        completed.add(key)
    except: pass
print(f'Completed: {len(completed)} evaluations')
for k in sorted(completed)[:10]:
    print(f'  {k[0][-30:]:30} {k[1]:20} {k[2]}')
"
```

Then rerun missing combinations:

```bash
# Run a single missing combination
python scripts/eval_cross_dataset.py \
  --model_path /scratch1/singhama/data/experiments/... \
  --dataset midterm \
  --task node_masking
```

### Analyze Results Programmatically

```python
import pandas as pd
import json
from pathlib import Path

# Load all results
results = []
for f in Path("/home1/eibl/gfm/prodigy/eval_results").rglob("*.json"):
    try:
        data = json.load(open(f))
        model_name = data["model_path"].split("/")[-2]
        results.append({
            "model": model_name,
            "dataset": data["dataset"],
            "task": data["task"],
            "test_acc": data["test_results"].get("accuracy", 0),
            "test_f1": data["test_results"].get("f1_score", 0),
        })
    except:
        pass

df = pd.DataFrame(results)

# Best models on each dataset
for dataset in df["dataset"].unique():
    print(f"\n{dataset}:")
    best = df[df["dataset"] == dataset].nlargest(3, "test_acc")
    print(best[["model", "task", "test_acc"]])

# Cross-task transfer analysis
print("\n=== Cross-task transfer ===")
nm_models = df[df["task"] == "node_masking"]["model"].unique()
print(f"Trained on node_masking, tested on link_prediction:")
lp_results = df[(df["task"] == "link_prediction") & (df["model"].isin(nm_models))]
print(lp_results[["model", "dataset", "test_acc"]])
```

---

## Expected Runtime

- **Per job**: 5-15 minutes (depends on dataset size and GPU)
- **All 90 jobs**: ~2-3 hours (with 40 parallel jobs)
- **Aggregation**: <1 minute

Total wall-clock time: ~4 hours from submission to results

---

## File Organization

Essential scripts:

| File | Purpose |
|------|---------|
| [scripts/eval_cross_dataset.py](scripts/eval_cross_dataset.py) | Main evaluation script |
| [scripts/generate_eval_jobs.py](scripts/generate_eval_jobs.py) | Generate job list and SLURM script |
| [scripts/aggregate_eval_results.py](scripts/aggregate_eval_results.py) | Summarize results |
| eval_cross_dataset.sbatch | Generated SLURM batch script |
| eval_jobs.txt | Generated job command list |

Generated during execution:

```
/scratch1/singhama/data/eval_results/
├── {dataset}/{task}/
│   └── eval_*.json (per evaluation)
├── summary.csv
└── summary.json
```

---

## Quick Reference

```bash
# One-liner: full pipeline
cd /scratch1/singhama/prodigy && \
python scripts/generate_eval_jobs.py && \
sbatch eval_cross_dataset.sbatch && \
echo "Jobs submitted! Monitor with: squeue -u \$USER"

# Monitor
watch -n 5 "squeue -u \$USER; echo; find /scratch1/singhama/data/eval_results -name '*.json' | wc -l"

# Results
python scripts/aggregate_eval_results.py --results_dir /scratch1/singhama/data/eval_results
```

---

## See Also

- [README.md](README.md) - Main documentation
- [CLAUDE.md](CLAUDE.md) - Code overview
- [experiments/trainer.py](experiments/trainer.py) - Model training/evaluation code
- [experiments/params.py](experiments/params.py) - Parameter definitions
