"""
Custom Dataset Template for PRODIGY

This template provides a starting point for loading your own graph dataset.
Modify the functions below to match your data format.

Usage:
1. Implement load_your_graph() to load your graph data
2. Update class_names with your actual class labels
3. Register in data_loader_wrapper.py and trainer.py
4. Run: python experiments/run_single_experiment.py --dataset custom_template ...
"""

import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from experiments.sampler import NeighborSampler
from .dataset import SubgraphDataset
from .dataloader import MulticlassTask, ParamSampler, BatchSampler, Collator
from .augment import get_aug


def load_your_graph(root):
    """
    MODIFY THIS: Load your graph data from files.

    Expected return: PyTorch Geometric Data object with:
        - x: Node features [num_nodes, feature_dim]
        - edge_index: Edge connections [2, num_edges]
        - y: Node labels [num_nodes]
        - num_nodes: Number of nodes

    Example implementations:
    """
    # Option 1: Load from CSV files
    # nodes_df = pd.read_csv(os.path.join(root, "nodes.csv"))
    # edges_df = pd.read_csv(os.path.join(root, "edges.csv"))
    #
    # x = torch.tensor(nodes_df[['feat1', 'feat2', ...]].values, dtype=torch.float)
    # edge_index = torch.tensor([edges_df['source'].values, edges_df['target'].values], dtype=torch.long)
    # y = torch.tensor(nodes_df['label'].values, dtype=torch.long)
    #
    # return Data(x=x, edge_index=edge_index, y=y, num_nodes=len(nodes_df))

    # Option 2: Load from numpy files
    # x = torch.from_numpy(np.load(os.path.join(root, "features.npy"))).float()
    # edge_index = torch.from_numpy(np.load(os.path.join(root, "edges.npy"))).long()
    # y = torch.from_numpy(np.load(os.path.join(root, "labels.npy"))).long()
    #
    # return Data(x=x, edge_index=edge_index, y=y, num_nodes=len(x))

    # Option 3: Load from PyTorch files
    # data = torch.load(os.path.join(root, "graph.pt"))
    # return data

    # Option 4: Create synthetic data (for testing)
    print("WARNING: Using synthetic data. Implement load_your_graph() for real data.")
    num_nodes = 1000
    num_edges = 5000
    num_classes = 10
    feature_dim = 768  # Match BERT dimension

    return Data(
        x=torch.randn(num_nodes, feature_dim),
        edge_index=torch.randint(0, num_nodes, (2, num_edges)),
        y=torch.randint(0, num_classes, (num_nodes,)),
        num_nodes=num_nodes
    )


def load_graph_with_text_features(root, bert, device):
    """
    MODIFY THIS: Load graph and add BERT-encoded text features.

    Use this if your nodes have text descriptions.
    """
    # Load basic graph structure
    # graph = load_your_graph(root)

    # Load or create node text descriptions
    # Option 1: From file
    # node_texts = pd.read_csv(os.path.join(root, "node_texts.csv"))['text'].tolist()

    # Option 2: From graph attributes
    # node_texts = [f"Node {i} description" for i in range(graph.num_nodes)]

    # Encode with BERT
    # embeddings = bert.get_sentence_embeddings(node_texts)
    # graph.x = embeddings

    # For now, use synthetic
    print("WARNING: Using synthetic text features. Implement load_graph_with_text_features().")
    return load_your_graph(root)


def get_custom_template_dataset(root, n_hop=2, bert=None, bert_device="cpu",
                               original_features=True, **kwargs):
    """
    Load custom dataset and create SubgraphDataset.

    Args:
        root: Root directory for dataset
        n_hop: Number of hops for neighbor sampling
        bert: BERT model for text encoding (if original_features=False)
        bert_device: Device for BERT
        original_features: If True, use graph.x; if False, encode with BERT

    Returns:
        SubgraphDataset object
    """
    # Cache path for preprocessed data
    cache_name = "custom_graph.pt"
    if not original_features and bert is not None:
        cache_name = f"custom_graph_bert_{bert.model_name}.pt"

    cache_path = os.path.join(root, cache_name)

    # Load or create graph
    if os.path.exists(cache_path):
        print(f"Loading cached graph from {cache_path}")
        graph = torch.load(cache_path)
    else:
        print(f"Creating graph and caching to {cache_path}")
        os.makedirs(root, exist_ok=True)

        if original_features or bert is None:
            graph = load_your_graph(root)
        else:
            graph = load_graph_with_text_features(root, bert, bert_device)

        # Cache for future use
        torch.save(graph, cache_path)

    print(f"Graph loaded: {graph.num_nodes} nodes, {graph.edge_index.shape[1]} edges")

    # Create neighbor sampler for subgraph extraction
    neighbor_sampler = NeighborSampler(graph, num_hops=n_hop)

    return SubgraphDataset(graph, neighbor_sampler)


def custom_template_task(root, split, label_set=None, split_labels=True,
                        train_cap=None, linear_probe=False):
    """
    Define few-shot learning task structure.

    Args:
        root: Data root directory
        split: 'train', 'val', or 'test'
        label_set: Specific label set to use (overrides split_labels)
        split_labels: If True, use different labels for train/val/test
        train_cap: Maximum training examples per class
        linear_probe: If True, use limited training data

    Returns:
        MulticlassTask object
    """
    # Load graph to get labels
    cache_path = os.path.join(root, "custom_graph.pt")
    if not os.path.exists(cache_path):
        # Create it if doesn't exist
        graph = load_your_graph(root)
        os.makedirs(root, exist_ok=True)
        torch.save(graph, cache_path)
    else:
        graph = torch.load(cache_path)

    labels = graph.y.numpy()
    num_classes = int(labels.max()) + 1

    print(f"Dataset has {num_classes} classes")

    if label_set is not None:
        # Use provided label set
        label_set = set(label_set)
    elif split_labels:
        # Meta-learning: different labels for each split
        # MODIFY THIS: Adjust the split ratios for your dataset
        all_labels = list(range(num_classes))

        # 60% train, 20% val, 20% test
        n_train = int(num_classes * 0.6)
        n_val = int(num_classes * 0.2)

        TRAIN_LABELS = all_labels[:n_train]
        VAL_LABELS = all_labels[n_train:n_train + n_val]
        TEST_LABELS = all_labels[n_train + n_val:]

        print(f"Split - Train: {len(TRAIN_LABELS)}, Val: {len(VAL_LABELS)}, Test: {len(TEST_LABELS)}")

        if split == "train":
            label_set = set(TRAIN_LABELS)
        elif split == "val":
            label_set = set(VAL_LABELS)
        elif split == "test":
            label_set = set(TEST_LABELS)
        else:
            raise ValueError(f"Invalid split: {split}")
    else:
        # Standard classification: all labels in all splits
        label_set = set(range(num_classes))

    # Handle train_cap for linear probing
    train_label = None
    if train_cap is not None and split == "train":
        train_label = labels.copy()
        for i in range(num_classes):
            idx = np.where(labels == i)[0]
            if len(idx) > train_cap:
                # Disable examples beyond train_cap
                disabled_idx = idx[train_cap:]
                train_label[disabled_idx] = -1 - i

    return MulticlassTask(labels, label_set, train_label, linear_probe)


def get_custom_template_dataloader(dataset, split, node_split, batch_size, n_way,
                                  n_shot, n_query, batch_count, root, bert,
                                  num_workers, aug, aug_test, split_labels,
                                  train_cap, linear_probe, label_set=None, **kwargs):
    """
    Create DataLoader for few-shot tasks.

    Args:
        dataset: SubgraphDataset object
        split: 'train', 'val', or 'test'
        batch_size: Number of tasks per batch
        n_way: Number of classes per task
        n_shot: Number of support examples per class
        n_query: Number of query examples per class
        batch_count: Total number of batches
        bert: BERT model for label embeddings
        num_workers: Number of data loading workers
        aug: Augmentation string (e.g., "ND0.5,NZ0.5")
        aug_test: Whether to augment test data
        split_labels: Whether to use different labels per split
        train_cap: Max training examples per class
        linear_probe: Whether doing linear probing
        label_set: Specific labels to use

    Returns:
        DataLoader object
    """
    # MODIFY THIS: Define your class names/descriptions
    class_names = [
        f"Class {i}" for i in range(10)  # Replace with actual class names
    ]

    # Example with real descriptions:
    # class_names = [
    #     "Computer Science",
    #     "Physics",
    #     "Mathematics",
    #     ...
    # ]

    # Get label embeddings from class names
    if bert is not None:
        label_embeddings = bert.get_sentence_embeddings(class_names)
    else:
        # Fallback: random embeddings (not recommended)
        label_embeddings = torch.randn(len(class_names), 768)

    # Create task sampler
    task = custom_template_task(root, split, label_set, split_labels,
                                train_cap, linear_probe)

    sampler = BatchSampler(
        batch_count,
        task,
        ParamSampler(batch_size, n_way, n_shot, n_query, 1),
        seed=42,
    )

    # Data augmentation
    if split == "train" or aug_test:
        aug_fn = get_aug(aug, dataset.graph.x)
    else:
        aug_fn = get_aug("")  # No augmentation

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        collate_fn=Collator(label_embeddings, aug=aug_fn)
    )

    return dataloader


# Example usage
if __name__ == "__main__":
    """Test the custom dataset loader."""
    from tqdm import tqdm
    from models.sentence_embedding import SentenceEmb

    # Setup
    root = "./data/custom_template"
    bert = SentenceEmb("sentence-transformers/all-mpnet-base-v2", device="cpu")

    # Load dataset
    print("Loading dataset...")
    dataset = get_custom_template_dataset(root, n_hop=2, bert=bert)

    # Create dataloader
    print("Creating dataloader...")
    dataloader = get_custom_template_dataloader(
        dataset, split="test",
        node_split="", batch_size=2,
        n_way=3, n_shot=3, n_query=5,
        batch_count=10, root=root,
        bert=bert, num_workers=0,
        aug="", aug_test=False,
        split_labels=True, train_cap=None,
        linear_probe=False
    )

    # Test iteration
    print("Testing dataloader...")
    for i, batch in enumerate(tqdm(dataloader)):
        print(f"Batch {i}: {len(batch)} items")
        if i >= 2:  # Test a few batches
            break

    print("Success! Dataset template is working.")
