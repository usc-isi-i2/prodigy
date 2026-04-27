#!/usr/bin/env python3
"""
Aggregate and summarize cross-dataset evaluation results.
"""

import json
import os
import sys
from pathlib import Path
import pandas as pd
from collections import defaultdict


def aggregate_results(eval_results_dir="./eval_results"):
    """
    Aggregate all evaluation results into a summary.
    """
    
    results_data = []
    
    # Find all evaluation JSON files
    for root, dirs, files in os.walk(eval_results_dir):
        for file in files:
            if file.endswith(".json"):
                result_path = os.path.join(root, file)
                try:
                    with open(result_path, 'r') as f:
                        result = json.load(f)
                    
                    # Extract key info
                    model_path = result.get("model_path", "unknown")
                    model_name = os.path.basename(os.path.dirname(model_path))
                    dataset = result.get("dataset", "unknown")
                    task = result.get("task", "unknown")
                    
                    # Extract metrics (adjust based on what your trainer returns)
                    test_results = result.get("test_results", {})
                    val_results = result.get("val_results", {})
                    
                    data_row = {
                        "model": model_name,
                        "model_path": model_path,
                        "dataset": dataset,
                        "task": task,
                        "test_results": test_results,
                        "val_results": val_results,
                        "timestamp": result.get("timestamp", "unknown"),
                    }
                    results_data.append(data_row)
                    
                except Exception as e:
                    print(f"ERROR reading {result_path}: {e}")
    
    if not results_data:
        print("No evaluation results found!")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(results_data)
    
    # Save full results
    summary_path = os.path.join(eval_results_dir, "summary.csv")
    df.to_csv(summary_path, index=False)
    print(f"Saved full results to {summary_path}")
    
    # Create pivot tables for different views
    
    # 1. Model performance on each dataset/task
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    
    # Organize by (model, dataset, task)
    summary_by_config = defaultdict(list)
    for _, row in df.iterrows():
        key = (row["model"], row["dataset"], row["task"])
        summary_by_config[key].append(row)
    
    # Print summary
    print(f"\n{'Model':<50} {'Dataset':<20} {'Task':<20}")
    print("-" * 90)
    for (model, dataset, task), rows in sorted(summary_by_config.items()):
        print(f"{model:<50} {dataset:<20} {task:<20}")
        if rows:
            test_res = rows[-1]["test_results"]  # Get latest
            print(f"  Test: {test_res}")
    
    # 2. Cross-dataset analysis
    print("\n" + "="*80)
    print("CROSS-DATASET EVALUATION RESULTS")
    print("="*80)
    
    for dataset in df["dataset"].unique():
        print(f"\nDataset: {dataset}")
        print("-" * 80)
        
        dataset_df = df[df["dataset"] == dataset]
        
        for task in dataset_df["task"].unique():
            task_df = dataset_df[dataset_df["task"] == task]
            print(f"  Task: {task}")
            
            for _, row in task_df.iterrows():
                model = row["model"]
                test_res = row["test_results"]
                print(f"    {model}: {test_res}")
    
    # 3. Save detailed JSON summary
    json_summary = {
        "total_evaluations": len(df),
        "models": len(df["model"].unique()),
        "datasets": len(df["dataset"].unique()),
        "tasks": len(df["task"].unique()),
        "results_by_model_dataset_task": {},
    }
    
    for (model, dataset, task), rows in summary_by_config.items():
        key = f"{model} | {dataset} | {task}"
        json_summary["results_by_model_dataset_task"][key] = {
            "test": rows[-1]["test_results"] if rows else None,
            "val": rows[-1]["val_results"] if rows else None,
        }
    
    json_summary_path = os.path.join(eval_results_dir, "summary.json")
    with open(json_summary_path, 'w') as f:
        json.dump(json_summary, f, indent=2)
    print(f"\nSaved JSON summary to {json_summary_path}")
    
    return df


def find_best_models(eval_results_dir="./eval_results", metric="accuracy", task=None, dataset=None):
    """Find best performing models."""
    
    results_data = []
    
    for root, dirs, files in os.walk(eval_results_dir):
        for file in files:
            if file.endswith(".json"):
                result_path = os.path.join(root, file)
                try:
                    with open(result_path, 'r') as f:
                        result = json.load(f)
                    
                    model_name = os.path.basename(os.path.dirname(result.get("model_path", "unknown")))
                    results_data.append({
                        "model": model_name,
                        "dataset": result.get("dataset", "unknown"),
                        "task": result.get("task", "unknown"),
                        "test_results": result.get("test_results", {}),
                    })
                except:
                    pass
    
    if not results_data:
        print("No results found")
        return
    
    df = pd.DataFrame(results_data)
    
    if task:
        df = df[df["task"] == task]
    if dataset:
        df = df[df["dataset"] == dataset]
    
    print("\n" + "="*80)
    print(f"TOP MODELS (metric={metric})")
    print("="*80)
    
    # This is a placeholder - adjust based on actual metrics in your results
    for _, row in df.iterrows():
        print(f"{row['model']:<50} {row['dataset']:<20} {row['task']:<20}")
        print(f"  Results: {row['test_results']}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Aggregate evaluation results")
    parser.add_argument("--results_dir", type=str, default="./eval_results",
                       help="Directory containing evaluation results")
    parser.add_argument("--best", action="store_true",
                       help="Show best models")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_dir):
        print(f"Results directory not found: {args.results_dir}")
        sys.exit(1)
    
    df = aggregate_results(args.results_dir)
    
    if args.best and df is not None:
        find_best_models(args.results_dir)
