# Created Files Summary

All files created for cross-dataset evaluation on cluster. Last updated: 2026-04-26

## 📂 File Manifest

### Core Evaluation Scripts

| File | Purpose | Usage |
|------|---------|-------|
| `scripts/eval_cross_dataset.py` | Main evaluation engine | `python eval_cross_dataset.py --model_path X --dataset Y --task Z` |
| `scripts/generate_eval_jobs.py` | Generate batch jobs | `python generate_eval_jobs.py` |
| `scripts/aggregate_eval_results.py` | Summarize results | `python aggregate_eval_results.py --results_dir DIR` |

### Generated on Cluster

| File | Created By | Purpose |
|------|-----------|---------|
| `eval_jobs.txt` | generate_eval_jobs.py | List of 90 evaluation commands |
| `eval_cross_dataset.sbatch` | generate_eval_jobs.py | SLURM batch submission script |

### Documentation

| File | Type | Contents |
|------|------|----------|
| `CROSS_DATASET_EVAL.md` | MD | Detailed guide (updated) |
| `EVAL_SETUP_SUMMARY.md` | MD | Executive summary & next steps |
| `scripts/EVAL_QUICKSTART.sh` | Bash | Quick reference commands |
| `scripts/check_eval_setup.py` | Python | Setup validator |
| `EVAL_FILES_MANIFEST.md` | MD | This file |

### Output Locations

| Location | Created By | Contents |
|----------|-----------|----------|
| `/scratch1/singhama/data/eval_results/` | eval_cross_dataset.py | All result JSONs |
| `/scratch1/singhama/data/eval_results/summary.csv` | aggregate_eval_results.py | Tabular results |
| `/scratch1/singhama/data/eval_results/summary.json` | aggregate_eval_results.py | Structured results |
| `/scratch1/singhama/logs/eval_*.log` | SLURM | Job logs |
| `/scratch1/singhama/logs/eval_*.err` | SLURM | Error logs |

## 🔄 Workflow Pipeline

```
generate_eval_jobs.py
    ↓
    Creates: eval_jobs.txt + eval_cross_dataset.sbatch
    ↓
sbatch eval_cross_dataset.sbatch
    ↓
    Submits 90 parallel jobs
    ↓
eval_cross_dataset.py × 90
    ↓
    Saves: eval_*.json files
    ↓
aggregate_eval_results.py
    ↓
    Creates: summary.csv + summary.json
    ↓
    Analysis & insights
```

## 📊 Configuration

### Evaluation Matrix
- **Models**: 15 (train2_* series)
- **Datasets**: 3 (midterm, covid19_twitter, ukr_rus_twitter)
- **Tasks**: 2 (node_masking, link_prediction)
- **Total Evaluations**: 90

### SLURM Configuration
```bash
--array=0-89%40           # 90 tasks, max 40 parallel
--nodes=1
--ntasks=1
--cpus-per-task=10
--gres=gpu:1              # 1 GPU per job
--time=4:00:00            # 4 hours per job
--mem=16GB
```

### Expected Resources
- **Total GPU hours**: ~90 hours (distributed across cluster)
- **Wall clock time**: 2-4 hours (with 40 parallel)
- **Disk space needed**: ~10-20 GB for results
- **Network I/O**: Low (datasets cached)

## 🎯 Quick Start Steps

1. **SSH to cluster**
   ```bash
   ssh user@cluster.edu
   ```

2. **Generate jobs**
   ```bash
   cd /scratch1/singhama/prodigy
   python scripts/generate_eval_jobs.py
   ```

3. **Submit batch**
   ```bash
   sbatch eval_cross_dataset.sbatch
   ```

4. **Monitor progress**
   ```bash
   watch -n 5 "squeue -u $USER"
   find /scratch1/singhama/data/eval_results -name "*.json" | wc -l
   ```

5. **Aggregate results**
   ```bash
   python scripts/aggregate_eval_results.py \
     --results_dir /scratch1/singhama/data/eval_results
   ```

## 📋 Model List

All 15 models included in evaluation:

1. `train2_midterm_nm_to_covid_nm_16_04_2026_10_07_00/state_dict`
2. `exp2_train2_midterm_nm_to_ukr_rus_nm_16_04_2026_10_32_38/checkpoint/state_dict_27000.ckpt`
3. `exp3_train2_covid_nm_to_ukr_rus_nm_16_04_2026_13_39_26/checkpoint/state_dict_30000.ckpt`
4. `exp4_train2_midterm_lp_to_covid_lp_16_04_2026_17_31_13/state_dict`
5. `exp5_train2_midterm_lp_to_ukr_rus_lp_16_04_2026_17_48_24/state_dict`
6. `exp6_train2_covid_lp_to_ukr_rus_lp_16_04_2026_17_24_43/state_dict`
7. `exp7_train2_midterm_nm_to_covid_lp_17_04_2026_16_35_55/state_dict`
8. `exp8_train2_midterm_nm_to_ukr_rus_lp_17_04_2026_16_36_31/state_dict`
9. `exp9_train2_covid_nm_to_ukr_rus_lp_17_04_2026_16_36_23/state_dict`
10. `exp10_train2_midterm_lp_to_covid_nm_22_04_2026_13_24_06/state_dict`
11. `exp11_train2_midterm_lp_to_ukr_rus_nm_22_04_2026_14_22_43/state_dict`
12. `exp12_train2_covid_lp_to_ukr_rus_nm_22_04_2026_14_22_38/state_dict`
13. `exp13_train2_covid_nm_to_midterm_nm_23_04_2026_13_24_00/checkpoint/state_dict_15000.ckpt`
14. `exp14_train2_ukr_rus_nm_to_midterm_nm_23_04_2026_13_26_26/checkpoint/state_dict_12000.ckpt`
15. `exp15_train2_ukr_rus_nm_to_covid_nm_23_04_2026_13_33_55/state_dict`

(Base directory: `/scratch1/singhama/data/experiments/`)

## ✅ Validation

Run this to verify setup:
```bash
cd /Users/philipp/projects/gfm/prodigy
python scripts/check_eval_setup.py
```

Should show all ✓ marks.

## 📖 Documentation References

| Document | Purpose |
|----------|---------|
| [CROSS_DATASET_EVAL.md](CROSS_DATASET_EVAL.md) | Full setup guide & troubleshooting |
| [EVAL_SETUP_SUMMARY.md](EVAL_SETUP_SUMMARY.md) | Executive summary |
| [scripts/EVAL_QUICKSTART.sh](scripts/EVAL_QUICKSTART.sh) | Copy-paste commands |
| [README.md](README.md) | Project overview |

## 🔗 Related Code

| File | Purpose |
|------|---------|
| `experiments/trainer.py` | TrainerFS class with do_eval() method |
| `experiments/params.py` | Parameter definitions |
| `data/data_loader_wrapper.py` | Dataset loading logic |
| `data/*.py` | Dataset-specific dataloaders |
| `models/*.py` | Model architectures |

## 📈 Expected Output Format

Each evaluation generates:

```json
{
  "model_path": "/scratch1/singhama/data/experiments/.../state_dict",
  "dataset": "midterm",
  "task": "node_masking",
  "timestamp": "2026-04-26T15:30:45.123456",
  "val_results": {
    "accuracy": 0.92,
    "f1_score": 0.89,
    "auc_roc": 0.94,
    "loss": 0.23
  },
  "test_results": {
    "accuracy": 0.89,
    "f1_score": 0.86,
    "auc_roc": 0.91,
    "loss": 0.25
  }
}
```

## 🆘 Support

For issues:
1. Check `/scratch1/singhama/logs/eval_*.err` for specific errors
2. Review [CROSS_DATASET_EVAL.md](CROSS_DATASET_EVAL.md) troubleshooting section
3. Run single evaluation to debug: `python eval_cross_dataset.py --model_path X --dataset Y --task Z`

---

**Last Updated**: 2026-04-26  
**Status**: ✅ Ready for cluster submission
