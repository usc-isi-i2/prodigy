import numpy as np
import random
import torch
import time

torch.multiprocessing.set_sharing_strategy("file_system")

import sys
import os

def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

sys.path.extend(os.path.join(os.path.dirname(__file__), "../../"))

from experiments.params import get_params
from experiments.trainer import TrainerFS

from data.data_loader_wrapper import get_dataset_wrap

import warnings

warnings.filterwarnings("ignore")


def seed_everything(params):
    """Apply the runner's historical deterministic seed setup."""
    if params['seed'] is None:
        return
    seed = params['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def load_dataset(params):
    """Load one dataset from resolved experiment parameters.

    Kept as a function so checkpoint sweeps can reuse the large immutable graph
    object instead of reading it once per checkpoint.
    """
    return get_dataset_wrap(
        root=params["root"],
        dataset=params["dataset"],
        force_cache=params["force_cache"],
        small_dataset=params["small_dataset"],
        invalidate_cache=None,
        original_features=params["original_features"],
        n_shot=params["n_shots"],
        n_query=params["n_query"],
        bert=None if params["original_features"] else params["bert_emb_model"],
        bert_device=params["device"],
        val_len_cap=params["val_len_cap"],
        test_len_cap=params["test_len_cap"],
        dataset_len_cap=params["dataset_len_cap"],
        n_way=params["n_way"],
        rel_sample_rand_seed=params["rel_sample_random_seed"],
        calc_ranks=params["calc_ranks"],
        kg_emb_model=params["kg_emb_model"] if params["kg_emb_model"] != "" else None,
        task_name=params["task_name"],
        shuffle_index=params["shuffle_index"],
        node_graph=params["task_name"] == "sn_neighbor_matching",
        csv_filename=params["csv_filename"],
        label_type=params["label_type"],
        max_users=params["max_users"],
        pkl_filename=params["facebook_pkl_filename"],
        facebook_edges_filename=params["facebook_edges_filename"],
        facebook_node_features_filename=params["facebook_node_features_filename"],
        facebook_data_source=params["facebook_data_source"],
        facebook_use_edge_features=params["facebook_use_edge_features"],
        facebook_edge_feature_columns=params["facebook_edge_feature_columns"],
        source_pkl_path=params["facebook_source_pkl_path"],
        facebook_embeddings_path=params["facebook_embeddings_path"],
        facebook_embedding_ids_path=params["facebook_embedding_ids_path"],
        facebook_text_emb_model=params["facebook_text_emb_model"],
        facebook_target_dim=params["facebook_target_dim"],
        facebook_filter_to_uk_ru=params["facebook_filter_to_uk_ru"],
        max_posts=params["facebook_max_posts"],
        n_hop=params["n_hop"],
        neighbor_sampling_hop_sizes=params["neighbor_sampling_hop_sizes"],
        neighbor_sampling_node_limit=params["neighbor_sampling_node_limit"],
        neighbor_matching_walk_hops=params["neighbor_matching_walk_hops"],
        graph_filename=params["graph_filename"],
        target_feature=params["target_feature"],
        target_feature_keep_in_x=params["target_feature_keep_in_x"],
        target_transform=params["target_transform"],
        feature_subset=params["feature_subset"],
        midterm_label_downsample=params["midterm_label_downsample"],
        edge_view=params["edge_view"],
        target_edge_view=params["target_edge_view"],
        edge_feature_subset=params["edge_feature_subset"],
        neighbor_sampling_strategy=params["neighbor_sampling_strategy"],
        neighbor_sampling_strata=params["neighbor_sampling_strata"],
        neighbor_matching_edge_split=params["neighbor_matching_edge_split"],
        seed=params["seed"],
    )

if __name__ == '__main__':
    torch.set_num_threads(4)

    params = get_params()

    seed_everything(params)

    if params["dataset"] in ["FB15K-237", "NELL", "ConceptNet", "Wiki"]:
        print("Using KG dataset - setting language model to sentence-transformers/all-mpnet-base-v2")
        params["bert_emb_model"] = "sentence-transformers/all-mpnet-base-v2"
    _log("Loading dataset...")
    datasets = load_dataset(params)

    _log("Dataset loaded. Initializing trainer...")
    trnr = TrainerFS(datasets, params)

    _log("Trainer initialized. Starting train/eval...")
    trnr.train()
