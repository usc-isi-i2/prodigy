#!/usr/bin/env python3
"""
Quick setup validator and info printer for cross-dataset evaluation.
"""

import os
import sys
from pathlib import Path


def check_setup():
    """Validate that everything is ready for evaluation."""
    
    checks = {
        "eval_cross_dataset.py": "/Users/philipp/projects/gfm/prodigy/scripts/eval_cross_dataset.py",
        "generate_eval_jobs.py": "/Users/philipp/projects/gfm/prodigy/scripts/generate_eval_jobs.py",
        "aggregate_eval_results.py": "/Users/philipp/projects/gfm/prodigy/scripts/aggregate_eval_results.py",
        "CROSS_DATASET_EVAL.md": "/Users/philipp/projects/gfm/prodigy/CROSS_DATASET_EVAL.md",
        "EVAL_QUICKSTART.sh": "/Users/philipp/projects/gfm/prodigy/scripts/EVAL_QUICKSTART.sh",
    }
    
    print("\n" + "="*80)
    print("CROSS-DATASET EVALUATION SETUP CHECK")
    print("="*80 + "\n")
    
    all_good = True
    for name, path in checks.items():
        exists = os.path.exists(path)
        status = "✓" if exists else "✗"
        print(f"{status} {name:<35} {path}")
        if not exists:
            all_good = False
    
    print("\n" + "="*80)
    print("QUICK START")
    print("="*80 + "\n")
    
    print("On your cluster, run:\n")
    print("  cd /home1/eibl/gfm/prodigy")
    print("  python scripts/generate_eval_jobs.py")
    print("  sbatch eval_cross_dataset.sbatch\n")
    
    print("="*80)
    print("EVALUATION CONFIGURATION")
    print("="*80 + "\n")
    
    print("Models:        15 trained models (train2 series)")
    print("Datasets:      3 (midterm, covid19_twitter, ukr_rus_twitter)")
    print("Tasks:         2 (node_masking, link_prediction)")
    print("Total jobs:    90 evaluations")
    print("Time/job:      ~5-15 minutes")
    print("Total time:    ~2-4 hours (with 40 parallel)")
    print()
    
    print("="*80)
    print("DOCUMENTATION")
    print("="*80 + "\n")
    
    print("📖 Full guide:       CROSS_DATASET_EVAL.md")
    print("⚡ Quick commands:   scripts/EVAL_QUICKSTART.sh")
    print("🔍 Main script:      scripts/eval_cross_dataset.py")
    print("📊 Aggregation:      scripts/aggregate_eval_results.py")
    print()
    
    print("="*80)
    print("NEXT STEPS")
    print("="*80 + "\n")
    
    print("1. SSH to cluster")
    print("2. cd /home1/eibl/gfm/prodigy")
    print("3. python scripts/generate_eval_jobs.py")
    print("4. sbatch eval_cross_dataset.sbatch")
    print("5. Monitor: squeue -u $USER")
    print("6. Aggregate: python scripts/aggregate_eval_results.py --results_dir /home1/eibl/gfm/prodigy/eval_results")
    print()
    
    if all_good:
        print("✓ All files created successfully!")
        return 0
    else:
        print("✗ Some files are missing - please check")
        return 1


if __name__ == "__main__":
    sys.exit(check_setup())
