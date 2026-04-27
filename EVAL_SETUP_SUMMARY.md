# ✅ Cross-Dataset Evaluation Pipeline - READY TO USE

Your evaluation pipeline is now fully set up and ready to run on the cluster!

## 📋 What You Have

You provided:
- **15 trained models** from cross-dataset transfer experiments (train2 series)
- **3 datasets** to evaluate on: midterm, covid19_twitter, ukr_rus_twitter  
- **2 tasks**: node_masking, link_prediction
- **Base directory**: /scratch1/singhama/data/experiments

## 🎯 What We Built

### 1. **eval_cross_dataset.py** - Core Evaluation Script
```bash
python scripts/eval_cross_dataset.py \
  --model_path <checkpoint_path> \
  --dataset <dataset_name> \
  --task <task_name> \
  --device 0 \
  --output_dir /scratch1/singhama/data/eval_results
```

**Features:**
- Loads model checkpoint from any path (state_dict or .ckpt)
- Evaluates on specified dataset/task
- Runs val and test set evaluation
- Saves results as JSON with metrics
- Handles GPU distribution automatically

### 2. **generate_eval_jobs.py** - Batch Job Creator
```bash
python scripts/generate_eval_jobs.py
```

**Creates:**
- `eval_jobs.txt` - List of all 90 evaluation commands
- `eval_cross_dataset.sbatch` - SLURM batch submission script

**Customizable:**
Edit the script to change which models, datasets, or tasks to evaluate

### 3. **aggregate_eval_results.py** - Results Aggregator
```bash
python scripts/aggregate_eval_results.py \
  --results_dir /scratch1/singhama/data/eval_results
```

**Outputs:**
- `summary.csv` - Tabular results (easy to load in pandas/Excel)
- `summary.json` - Structured results with metadata
- Console printout with statistics and rankings

### 4. **CROSS_DATASET_EVAL.md** - Complete Documentation
- Step-by-step setup guide
- Configuration options
- Troubleshooting
- Advanced usage
- Performance tips

### 5. **EVAL_QUICKSTART.sh** - Quick Reference
Copy-paste commands for common operations

## 🚀 Quick Start (3 Steps)

### Step 1: Generate Jobs (on cluster)
```bash
cd /scratch1/singhama/prodigy
python scripts/generate_eval_jobs.py
```

This creates two files:
- `eval_jobs.txt` - 90 evaluation commands (review if needed)
- `eval_cross_dataset.sbatch` - SLURM submission script

### Step 2: Submit to Cluster
```bash
sbatch eval_cross_dataset.sbatch
```

SLURM will:
- Create an array job with 90 tasks
- Run up to 40 jobs in parallel
- Automatically distribute GPU IDs (0-3)
- Log output to `/scratch1/singhama/logs/eval_*.log`

### Step 3: Monitor & Aggregate
```bash
# Monitor progress
squeue -u $USER
watch -n 5 "find /scratch1/singhama/data/eval_results -name '*.json' | wc -l"

# After all jobs complete (2-4 hours)
python scripts/aggregate_eval_results.py --results_dir /scratch1/singhama/data/eval_results
```

## 📊 Evaluation Matrix

Your 15 × 3 × 2 = **90 evaluations** cover:

| Combination | Examples |
|---|---|
| Same task transfer | train2_midterm_nm → eval on covid (nm) |
| Cross-task transfer | train2_midterm_nm → eval on covid (lp) |
| Backward transfer | train2_covid → eval on midterm |
| Full cross-domain | All 15 models on all 3 datasets × 2 tasks |

## 📁 Output Structure

```
/scratch1/singhama/data/eval_results/
├── midterm/
│   ├── node_masking/
│   │   └── eval_midterm_node_masking_26_04_2026_15_30_45.json
│   └── link_prediction/
│       └── eval_midterm_link_prediction_26_04_2026_15_35_22.json
├── covid19_twitter/
│   ├── node_masking/
│   └── link_prediction/
├── ukr_rus_twitter/
│   ├── node_masking/
│   └── link_prediction/
├── summary.csv
└── summary.json
```

Each JSON file contains:
```json
{
  "model_path": "...",
  "dataset": "midterm",
  "task": "node_masking",
  "timestamp": "2026-04-26T...",
  "val_results": {"accuracy": 0.92, "f1_score": 0.89, ...},
  "test_results": {"accuracy": 0.89, "f1_score": 0.86, ...}
}
```

## ⏱️ Expected Timeline

| Step | Time |
|---|---|
| Generate jobs | < 1 min |
| SLURM submission | < 1 min |
| Evaluations (90 jobs, 40 parallel) | 2-4 hours |
| Aggregation | < 1 min |
| **Total** | **~3-5 hours** |

## 🎚️ Customization

### Run Specific Combinations
Edit `scripts/generate_eval_jobs.py`:
```python
MODELS = [
    "/path/to/model1",
    "/path/to/model2",
    # Add/remove as needed
]

DATASETS = ["midterm", "covid19_twitter"]  # Skip ukr_rus_twitter
TASKS = ["node_masking"]  # Only node masking
```

### Adjust SLURM Settings
```bash
# More parallel jobs
sbatch --array=0-89%80 eval_cross_dataset.sbatch

# Different time limit
sbatch --time=6:00:00 eval_cross_dataset.sbatch

# Different memory
sbatch --mem=32GB eval_cross_dataset.sbatch
```

### Test Single Evaluation First
```bash
python scripts/eval_cross_dataset.py \
  --model_path /scratch1/singhama/data/experiments/train2_midterm_nm_to_covid_nm_16_04_2026_10_07_00/state_dict \
  --dataset covid19_twitter \
  --task node_masking \
  --output_dir /tmp/test
```

## 🔧 Troubleshooting

| Problem | Solution |
|---|---|
| Jobs stuck in PENDING | Check cluster resources: `sinfo` |
| Out of memory | Use `--batch_size 2 --dataset_len_cap 5000` |
| Model checkpoint not found | Verify path: `ls -la /path/to/model` |
| Dataset loading error | Try `--force_cache false` |
| Need to rerun failed jobs | Check `summary.csv`, manually run missing combinations |

## 📈 Analysis Examples

### Load and analyze results in Python
```python
import pandas as pd
import json

# Load summary
df = pd.read_csv("/scratch1/singhama/data/eval_results/summary.csv")

# Best models on each dataset
for dataset in df["dataset"].unique():
    best = df[df["dataset"] == dataset].nlargest(3, "test_accuracy")
    print(f"\n{dataset}:")
    print(best[["model", "task", "test_accuracy"]])

# Cross-task transfer analysis
print("\nNode masking → Link prediction transfer:")
nm_results = df[df["task"] == "node_masking"]
print(nm_results[["model", "dataset", "test_accuracy"]])
```

## 📚 Documentation

- **[CROSS_DATASET_EVAL.md](CROSS_DATASET_EVAL.md)** - Full guide with examples
- **[scripts/EVAL_QUICKSTART.sh](scripts/EVAL_QUICKSTART.sh)** - Copy-paste commands
- **[scripts/eval_cross_dataset.py](scripts/eval_cross_dataset.py)** - Source code with inline docs
- **[experiments/trainer.py](../experiments/trainer.py)** - Evaluation implementation

## ✅ Validation Checklist

Before submitting:
- [ ] Connected to cluster
- [ ] Dataset paths are correct (`/scratch1/singhama/data`)
- [ ] CUDA available on nodes (`nvidia-smi`)
- [ ] Conda environment set up (see sbatch script)
- [ ] Sufficient disk space for results (~10GB for 90 JSONs)

## 🎓 What This Measures

For each model × dataset × task combination, you get:

**Standard Classification Metrics:**
- Accuracy (% correct predictions)
- F1 Score (harmonic mean of precision/recall)
- AUC-ROC (area under ROC curve)

**Optional Metrics** (if enabled in trainer):
- MRR (Mean Reciprocal Rank)
- HITS@K (Hit rate at K)
- Loss values

These metrics tell you:
- ✅ How well models transfer between datasets
- ✅ Cross-task generalization (nm→lp)
- ✅ Backward transfer (target→source)
- ✅ Full cross-domain capabilities

## 🎯 Next Action

**SSH to cluster and run:**

```bash
cd /scratch1/singhama/prodigy
python scripts/generate_eval_jobs.py
sbatch eval_cross_dataset.sbatch
echo "Submitted $(grep -c 'python' eval_jobs.txt) evaluation jobs!"
```

Monitor with:
```bash
watch -n 5 "squeue -u $USER; echo '---'; find /scratch1/singhama/data/eval_results -name '*.json' | wc -l"
```

---

**Questions?** Check [CROSS_DATASET_EVAL.md](CROSS_DATASET_EVAL.md) or [scripts/EVAL_QUICKSTART.sh](scripts/EVAL_QUICKSTART.sh)
