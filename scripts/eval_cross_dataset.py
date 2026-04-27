#!/usr/bin/env python3
"""
Cross-dataset evaluation script.
Loads a trained model and evaluates it on a specified dataset and task.

Usage:
    python eval_cross_dataset.py --model_path <path> --dataset <dataset> --task <task> [options]
"""

import sys
import os
import json
import torch
import argparse
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.params import get_params, str2bool
from experiments.trainer import TrainerFS
from data.data_loader_wrapper import get_dataset_wrap


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def setup_eval_params(model_config_dict, dataset, task, device=0):
    """
    Create evaluation parameters by merging model config with eval specs.
    """
    # Start with model config
    params = model_config_dict.copy()
    
    # Override for evaluation
    params["eval_only"] = True
    params["eval_test_before_train"] = True
    params["eval_val_before_train"] = True
    params["dataset"] = dataset
    params["task_name"] = task
    params["device"] = device
    params["exp_name"] = f"eval_{dataset}_{task}_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}"
    
    # Disable wandb logging for eval
    os.environ["WANDB_MODE"] = "disabled"
    
    return params


def load_model_config(model_dir):
    """Load the config.json from the model directory."""
    config_path = os.path.join(os.path.dirname(model_dir), "config.json")
    
    if not os.path.exists(config_path):
        _log(f"WARNING: config.json not found at {config_path}")
        _log("Using default parameters - results may be incorrect!")
        return {}
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    _log(f"Loaded config from {config_path}")
    return config


def main():
    parser = argparse.ArgumentParser(description="Cross-dataset model evaluation")
    parser.add_argument("--model_path", required=True, type=str, 
                       help="Path to model checkpoint (state_dict or .ckpt file)")
    parser.add_argument("--dataset", required=True, type=str,
                       help="Dataset to evaluate on (e.g., midterm, covid19_twitter, ukr_rus_twitter)")
    parser.add_argument("--task", required=True, type=str,
                       help="Task name (e.g., node_masking, link_prediction, classification)")
    parser.add_argument("--root", type=str, default="/home1/eibl/gfm/prodigy/FSdatasets",
                       help="Root directory for datasets")
    parser.add_argument("--device", type=int, default=0,
                       help="GPU device ID")
    parser.add_argument("--output_dir", type=str, default="./eval_results",
                       help="Directory to save evaluation results")
    parser.add_argument("--force_cache", type=str2bool, default=False,
                       help="Use cached dataset if available")
    parser.add_argument("--batch_size", type=int, default=5,
                       help="Batch size for evaluation")
    parser.add_argument("--dataset_len_cap", type=int, default=10000,
                       help="Number of samples to evaluate")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.model_path):
        _log(f"ERROR: Model path does not exist: {args.model_path}")
        sys.exit(1)
    
    _log("=" * 80)
    _log(f"Cross-Dataset Evaluation")
    _log(f"Model: {args.model_path}")
    _log(f"Dataset: {args.dataset}, Task: {args.task}")
    _log("=" * 80)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model config
    model_dir = os.path.dirname(args.model_path)
    model_config = load_model_config(args.model_path)
    
    # Setup evaluation parameters
    try:
        eval_params = setup_eval_params(model_config, args.dataset, args.task, device=args.device)
        eval_params["root"] = args.root
        eval_params["force_cache"] = args.force_cache
        eval_params["batch_size"] = args.batch_size
        eval_params["dataset_len_cap"] = args.dataset_len_cap
        eval_params["n_way"] = model_config.get("n_way", 3)
        eval_params["n_shots"] = model_config.get("n_shots", 3)
        eval_params["n_query"] = model_config.get("n_query", 24)
        eval_params["state_dir"] = os.path.join(args.output_dir, "state")
        eval_params["log_dir"] = os.path.join(args.output_dir, "log")
        
        _log(f"Loading dataset: {args.dataset}")
        
        # Load dataset
        datasets = get_dataset_wrap(
            root=eval_params["root"],
            dataset=eval_params["dataset"],
            force_cache=eval_params["force_cache"],
            small_dataset=eval_params.get("small_dataset", False),
            invalidate_cache=None,
            original_features=eval_params.get("original_features", False),
            n_shot=eval_params["n_shots"],
            n_query=eval_params["n_query"],
            bert=None if eval_params.get("original_features", False) else eval_params.get("bert_emb_model", "sentence-transformers/all-mpnet-base-v2"),
            bert_device=eval_params["device"],
            val_len_cap=eval_params.get("val_len_cap"),
            test_len_cap=eval_params.get("test_len_cap"),
            dataset_len_cap=eval_params["dataset_len_cap"],
            n_way=eval_params["n_way"],
            rel_sample_rand_seed=eval_params.get("rel_sample_random_seed"),
            calc_ranks=eval_params.get("calc_ranks", False),
            kg_emb_model=eval_params.get("kg_emb_model", "") if eval_params.get("kg_emb_model", "") != "" else None,
            task_name=eval_params["task_name"],
            shuffle_index=eval_params.get("shuffle_index", False),
            node_graph=eval_params["task_name"] == "sn_neighbor_matching",
            csv_filename=eval_params.get("csv_filename", "twitter_data.csv"),
            label_type=eval_params.get("label_type", "verified"),
            max_users=eval_params.get("max_users"),
            pkl_filename=eval_params.get("facebook_pkl_filename", "facebook_2022-06-24_2022-06-25_part1.pkl"),
            facebook_edges_filename=eval_params.get("facebook_edges_filename", "facebook_2022-06-24_2022-06-25_part1_edges.csv"),
            facebook_node_features_filename=eval_params.get("facebook_node_features_filename", "facebook_2022-06-24_2022-06-25_part1_node_features.csv"),
            facebook_data_source=eval_params.get("facebook_data_source", "csv"),
            facebook_use_edge_features=eval_params.get("facebook_use_edge_features", False),
            facebook_edge_feature_columns=eval_params.get("facebook_edge_feature_columns", ""),
            source_pkl_path=eval_params.get("facebook_source_pkl_path", ""),
            facebook_embeddings_path=eval_params.get("facebook_embeddings_path", ""),
            facebook_embedding_ids_path=eval_params.get("facebook_embedding_ids_path", ""),
            facebook_text_emb_model=eval_params.get("facebook_text_emb_model", ""),
            facebook_target_dim=eval_params.get("facebook_target_dim", 768),
            facebook_filter_to_uk_ru=eval_params.get("facebook_filter_to_uk_ru", True),
            max_posts=eval_params.get("facebook_max_posts"),
            n_hop=eval_params.get("n_hop", 2),
            graph_filename=eval_params.get("graph_filename", ""),
            midterm_feature_subset=eval_params.get("midterm_feature_subset", ""),
            midterm_label_downsample=eval_params.get("midterm_label_downsample", 1),
            midterm_edge_view=eval_params.get("midterm_edge_view", ""),
            midterm_target_edge_view=eval_params.get("midterm_target_edge_view", ""),
            midterm_edge_feature_subset=eval_params.get("midterm_edge_feature_subset", ""),
            seed=eval_params.get("seed"),
        )
        
        _log("Dataset loaded. Initializing trainer...")
        
        # Initialize trainer
        trainer = TrainerFS(datasets, eval_params)
        
        # Load model checkpoint
        _log(f"Loading model checkpoint: {args.model_path}")
        trainer.load_checkpoint(args.model_path)
        
        # Run evaluation
        _log("Running evaluation on val set...")
        val_results = trainer.do_eval(trainer.val_dataloader, split_name="val")
        
        _log("Running evaluation on test set...")
        test_results = trainer.do_eval(trainer.test_dataloader, split_name="test")
        
        # Save results
        result_file = os.path.join(
            args.output_dir, 
            f"eval_{args.dataset}_{args.task}_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.json"
        )
        
        results = {
            "model_path": args.model_path,
            "dataset": args.dataset,
            "task": args.task,
            "timestamp": datetime.now().isoformat(),
            "val_results": val_results,
            "test_results": test_results,
        }
        
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        _log(f"Results saved to {result_file}")
        _log("=" * 80)
        _log(f"Val Results: {val_results}")
        _log(f"Test Results: {test_results}")
        _log("=" * 80)
        
        return 0
        
    except Exception as e:
        _log(f"ERROR during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
