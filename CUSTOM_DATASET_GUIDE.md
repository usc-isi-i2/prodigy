# Loading Custom Datasets in PRODIGY - Complete Guide

This guide explains how to load your own custom graph dataset for inference tasks in PRODIGY.

## Overview

PRODIGY uses a multi-layered data loading approach:
1. **Base Dataset**: Contains the full graph structure and features
2. **Subgraph Dataset**: Samples subgraphs around nodes/edges
3. **Task Definition**: Defines few-shot learning tasks (n-way k-shot)
4. **DataLoader**: Batches tasks and creates prompt graphs

## Quick Start: Minimal Custom Dataset

### Step 1: Prepare Your Graph Data

Your custom dataset needs the following components in PyTorch Geometric format:

```python
import torch
from torch_geometric.data import Data

# Create your graph
graph = Data(
    # Node features: [num_nodes, feature_dim]
    x=torch.randn(1000, 768),  # 1000 nodes, 768-dim features (BERT-like)

    # Edge connections: [2, num_edges]
    edge_index=torch.tensor([[0, 1, 2, ...], [1, 2, 3, ...]]),

    # Node labels: [num_nodes]
    y=torch.randint(0, 10, (1000,)),  # 10 classes

    # Optional: Edge attributes
    edge_attr=torch.randn(num_edges, edge_feature_dim),

    num_nodes=1000
)
```

### Step 2: Create Custom Dataset Module

Create a new file `data/my_custom.py`:

```python
import os
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from experiments.sampler import NeighborSampler
from .dataset import SubgraphDataset
from .dataloader import MulticlassTask, ParamSampler, BatchSampler, Collator
from .augment import get_aug


def get_my_custom_dataset(root, n_hop=2, bert=None, bert_device="cpu", **kwargs):
    """
    Load your custom graph dataset.

    Args:
        root: Root directory for data
        n_hop: Number of hops for subgraph sampling
        bert: Sentence embedding model (optional, for text features)
        bert_device: Device for BERT processing

    Returns:
        SubgraphDataset object
    """
    cache_path = os.path.join(root, "my_custom_graph.pt")

    # Load or create your graph
    if os.path.exists(cache_path):
        graph = torch.load(cache_path)
    else:
        # Option 1: Load from your own format
        graph = load_my_graph_format(root)

        # Option 2: Create graph from scratch
        # graph = create_graph_from_csv(root)

        # Option 3: Use text features with BERT
        if bert is not None:
            graph = add_text_features(graph, bert, bert_device)

        # Cache for future use
        torch.save(graph, cache_path)

    # Create neighbor sampler for subgraph extraction
    neighbor_sampler = NeighborSampler(graph, num_hops=n_hop)

    return SubgraphDataset(graph, neighbor_sampler)


def load_my_graph_format(root):
    """
    Load graph from your custom format.
    Replace this with your actual data loading logic.
    """
    # Example: Load from files
    # nodes = pd.read_csv(os.path.join(root, "nodes.csv"))
    # edges = pd.read_csv(os.path.join(root, "edges.csv"))

    # Create PyG Data object
    graph = Data(
        x=torch.randn(1000, 768),  # Replace with actual features
        edge_index=torch.randint(0, 1000, (2, 5000)),  # Replace with actual edges
        y=torch.randint(0, 10, (1000,)),  # Replace with actual labels
        num_nodes=1000
    )

    return graph


def add_text_features(graph, bert, device):
    """
    Add BERT-encoded text features to graph nodes.
    """
    # Example: if you have node texts
    node_texts = ["Node description " + str(i) for i in range(graph.num_nodes)]

    # Encode with BERT
    embeddings = bert.get_sentence_embeddings(node_texts)
    graph.x = embeddings

    return graph


def my_custom_task(split, label_set=None, split_labels=True, train_cap=3, linear_probe=False):
    """
    Define task structure for few-shot learning.

    Args:
        split: 'train', 'val', or 'test'
        label_set: Set of labels to use for this split
        split_labels: Whether to split labels across train/val/test
        train_cap: Max examples per class in training
        linear_probe: Whether doing linear probing
    """
    # Load graph to get labels
    graph = torch.load("path/to/my_custom_graph.pt")
    label = graph.y.numpy()

    if split_labels:
        # Meta-learning setting: different labels for train/val/test
        # Define your label splits
        TRAIN_LABELS = list(range(0, 6))  # Classes 0-5 for training
        VAL_LABELS = list(range(6, 8))    # Classes 6-7 for validation
        TEST_LABELS = list(range(8, 10))  # Classes 8-9 for testing

        if split == "train":
            label_set = set(TRAIN_LABELS)
        elif split == "val":
            label_set = set(VAL_LABELS)
        elif split == "test":
            label_set = set(TEST_LABELS)
    else:
        # Standard classification: same labels everywhere
        if label_set is None:
            label_set = set(range(10))

    return MulticlassTask(label, label_set, train_label=None, linear_probe=linear_probe)


def get_my_custom_dataloader(dataset, split, node_split, batch_size, n_way, n_shot,
                              n_query, batch_count, root, bert, num_workers,
                              aug, aug_test, split_labels, train_cap, linear_probe,
                              label_set=None, **kwargs):
    """
    Create DataLoader for few-shot learning tasks.

    Args:
        dataset: SubgraphDataset object
        split: 'train', 'val', or 'test'
        batch_size: Number of tasks per batch
        n_way: Number of classes per task
        n_shot: Number of support examples per class
        n_query: Number of query examples per class
        batch_count: Total number of batches
        bert: Sentence embedding model for label embeddings
        aug: Augmentation string (e.g., "ND0.5,NZ0.5")
        aug_test: Whether to augment test data
    """
    # Get label embeddings (class descriptions)
    # Option 1: Text descriptions
    class_names = ["Class A", "Class B", "Class C", ...]  # Your class names
    label_embeddings = bert.get_sentence_embeddings(class_names)

    # Option 2: Learned embeddings (if you don't have text)
    # label_embeddings = torch.randn(num_classes, 768)

    # Create task sampler
    sampler = BatchSampler(
        batch_count,
        my_custom_task(split, label_set, split_labels, train_cap, linear_probe),
        ParamSampler(batch_size, n_way, n_shot, n_query, 1),
        seed=42,
    )

    # Data augmentation
    if split == "train" or aug_test:
        aug = get_aug(aug, dataset.graph.x)
    else:
        aug = get_aug("")

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        collate_fn=Collator(label_embeddings, aug=aug)
    )

    return dataloader
```

### Step 3: Register Dataset in Wrapper

Edit `data/data_loader_wrapper.py`:

```python
def get_dataset_wrap(root, dataset, **kwargs):
    if dataset == "arxiv":
        from data.arxiv import get_arxiv_dataset
        return get_arxiv_dataset(root=os.path.join(root, "arxiv"), **kwargs)
    # ... existing datasets ...

    # Add your custom dataset
    elif dataset == "my_custom":
        from data.my_custom import get_my_custom_dataset
        return get_my_custom_dataset(root=os.path.join(root, "my_custom"), **kwargs)

    else:
        raise NotImplementedError
```

### Step 4: Register DataLoader in Trainer

Edit `experiments/trainer.py` in the `_build_dataloaders` method (around line 218):

```python
def _build_dataloaders(self, dataset, dataset_name):
    # ... existing code ...

    if dataset_name == "arxiv":
        from data.arxiv import get_arxiv_dataloader
        get_dataloader = get_arxiv_dataloader
    # ... existing datasets ...
    elif dataset_name == "my_custom":
        from data.my_custom import get_my_custom_dataloader
        get_dataloader = get_my_custom_dataloader
    else:
        raise NotImplementedError

    # ... rest of the method ...
```

### Step 5: Run Inference

Now you can run inference with your custom dataset:

```bash
python experiments/run_single_experiment.py \
  --dataset my_custom \
  --root ./data \
  --layers S2,U,M \
  -way 3 -shot 3 -qry 3 \
  -pretrained path/to/checkpoint.ckpt \
  --eval_only True \
  --device 0
```

## Advanced: Custom Graph Structure

### For Knowledge Graph-style Data (Edge Prediction)

If you have a knowledge graph with (head, relation, tail) triples:

```python
from .dataset import KGSubgraphDataset
from data.load_kg_dataset import SubgraphFewshotDatasetWithTextFeats

def get_my_kg_dataset(root, name, n_hop=2, bert=None, **kwargs):
    """
    Load custom knowledge graph dataset.
    """
    # Create KG dataset object
    class MyKGDataset:
        def __init__(self, root):
            # Load your triples
            self.graph = self.load_graph(root)
            self.dataset = name
            self.hop = n_hop
            self.kind = "union"

            # Maps for entities and relations
            self.id2entity = {i: f"entity_{i}" for i in range(self.graph.num_nodes)}
            self.id2relation = {i: f"relation_{i}" for i in range(num_relations)}

            # Optional: pretrained embeddings
            self.pretrained_embeddings = None
            self.disk_features = None

        def load_graph(self, root):
            # Load edge_index: [2, num_triples]
            # edge_attr: [num_triples] - relation IDs
            return Data(
                edge_index=torch.tensor([[0, 1], [1, 2]]),
                edge_attr=torch.tensor([0, 1]),
                num_nodes=1000
            )

    dataset = MyKGDataset(root)

    # Add text features if needed
    if bert is not None:
        entity_texts = list(dataset.id2entity.values())
        dataset.text_feats = {
            text: emb for text, emb in
            zip(entity_texts, bert.get_sentence_embeddings(entity_texts))
        }

    # Create neighbor sampler
    from experiments.sampler import NeighborSamplerCacheAdj
    graph_ns = Data(edge_index=dataset.graph.edge_index,
                    num_nodes=dataset.graph.num_nodes)
    neighbor_sampler = NeighborSamplerCacheAdj(
        os.path.join(root, f"{name}_adj.pt"),
        graph_ns,
        n_hop
    )

    return KGSubgraphDataset(dataset, neighbor_sampler, "new",
                             node_graph=kwargs.get("node_graph", False))
```

## Data Format Examples

### Example 1: CSV Files

```python
def load_from_csv(root):
    import pandas as pd

    # Load nodes
    nodes_df = pd.read_csv(os.path.join(root, "nodes.csv"))
    # Expected columns: node_id, feature_1, feature_2, ..., label

    # Load edges
    edges_df = pd.read_csv(os.path.join(root, "edges.csv"))
    # Expected columns: source, target, (optional: edge_type, weight)

    # Create feature matrix
    feature_cols = [c for c in nodes_df.columns if c.startswith('feature_')]
    x = torch.tensor(nodes_df[feature_cols].values, dtype=torch.float)

    # Create edge index
    edge_index = torch.tensor([
        nodes_df['node_id'].tolist(),
        edges_df['target'].tolist()
    ], dtype=torch.long)

    # Create labels
    y = torch.tensor(nodes_df['label'].values, dtype=torch.long)

    return Data(x=x, edge_index=edge_index, y=y, num_nodes=len(nodes_df))
```

### Example 2: NetworkX Graph

```python
def load_from_networkx(G):
    import networkx as nx
    from torch_geometric.utils import from_networkx

    # Convert NetworkX to PyG
    data = from_networkx(G)

    # Add features if not present
    if not hasattr(data, 'x'):
        data.x = torch.eye(data.num_nodes)  # One-hot encoding

    # Add labels if not present
    if not hasattr(data, 'y'):
        # Extract from node attributes
        labels = [G.nodes[i].get('label', 0) for i in range(data.num_nodes)]
        data.y = torch.tensor(labels)

    return data
```

### Example 3: Text Nodes (Like arXiv)

```python
def load_with_text_features(root, bert, device):
    import json

    # Load node texts
    with open(os.path.join(root, "node_texts.json")) as f:
        node_data = json.load(f)

    # Extract texts and labels
    texts = [item['text'] for item in node_data]
    labels = [item['label'] for item in node_data]

    # Encode with BERT
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(bert, device=device)
    embeddings = model.encode(texts, show_progress_bar=True,
                             convert_to_tensor=True)

    # Load edges
    edges = torch.load(os.path.join(root, "edges.pt"))

    return Data(
        x=embeddings.cpu(),
        edge_index=edges,
        y=torch.tensor(labels),
        num_nodes=len(texts)
    )
```

## Key Configuration Parameters

When running experiments with your custom dataset:

```bash
# Basic inference
python experiments/run_single_experiment.py \
  --dataset my_custom \           # Your dataset name
  --root ./data \                 # Data root directory
  --layers S2,U,M \              # Model architecture
  -way 5 \                       # 5-way classification
  -shot 3 \                      # 3 support examples per class
  -qry 10 \                      # 10 query examples per class
  -pretrained checkpoint.ckpt \  # Pretrained model
  --eval_only True \             # Inference only
  --device 0                     # GPU device

# With text features
python experiments/run_single_experiment.py \
  --dataset my_custom \
  --root ./data \
  --original_features False \     # Use BERT features
  -bert roberta-base \           # BERT model
  -pretrained checkpoint.ckpt \
  --eval_only True

# With data augmentation (for robustness)
python experiments/run_single_experiment.py \
  --dataset my_custom \
  -aug ND0.3,NZ0.3 \            # Node drop 30%, Zero 30%
  -aug_test True \               # Augment test data too
  --eval_only True
```

## Common Issues and Solutions

### Issue 1: Feature Dimension Mismatch

**Problem**: Pretrained model expects 768-dim features, but your data has different dimensions.

**Solution**:
```python
# Option 1: Use BERT to get 768-dim features
embeddings = bert.encode(node_texts)  # Returns 768-dim

# Option 2: Add projection layer
def add_projection(graph, target_dim=768):
    current_dim = graph.x.shape[1]
    projection = torch.nn.Linear(current_dim, target_dim)
    graph.x = projection(graph.x)
    return graph

# Option 3: Update model's input_dim parameter
--input_dim 300  # Match your feature dimension
```

### Issue 2: Label Mismatch

**Problem**: Different number of classes than pretrained model.

**Solution**: The model should adapt automatically for few-shot learning, but ensure:
```python
# In your task definition
label_set = set(range(num_classes))  # All your classes

# When running
-way 3  # Should be <= num_classes
```

### Issue 3: Out of Memory

**Solution**:
```bash
# Reduce batch size
-bs 1

# Reduce query examples
-qry 3

# Use smaller model
--layers S,U,M  # Instead of S2,U,M

# Reduce dataset size
-test_cap 100  # Only use 100 batches
```

## Complete Working Example

Here's a complete minimal example in `data/my_simple.py`:

```python
import os
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from experiments.sampler import NeighborSampler
from .dataset import SubgraphDataset
from .dataloader import MulticlassTask, ParamSampler, BatchSampler, Collator
from .augment import get_aug


def get_simple_dataset(root, **kwargs):
    """Simple example dataset."""
    cache_path = os.path.join(root, "simple_graph.pt")

    if os.path.exists(cache_path):
        graph = torch.load(cache_path)
    else:
        # Create a simple synthetic graph
        num_nodes = 1000
        num_classes = 10

        graph = Data(
            x=torch.randn(num_nodes, 768),  # Random features
            edge_index=torch.randint(0, num_nodes, (2, 5000)),
            y=torch.randint(0, num_classes, (num_nodes,)),
            num_nodes=num_nodes
        )
        torch.save(graph, cache_path)

    neighbor_sampler = NeighborSampler(graph, num_hops=2)
    return SubgraphDataset(graph, neighbor_sampler)


def get_simple_dataloader(dataset, split, batch_size, n_way, n_shot, n_query,
                          batch_count, bert, num_workers, aug, aug_test, **kwargs):
    """Simple dataloader."""
    # Simple label embeddings
    label_embeddings = torch.randn(10, 768)

    # Task definition
    graph = dataset.graph
    label = graph.y.numpy()
    label_set = set(range(10))
    task = MulticlassTask(label, label_set, None, False)

    # Sampler
    sampler = BatchSampler(
        batch_count, task,
        ParamSampler(batch_size, n_way, n_shot, n_query, 1),
        seed=42
    )

    # Augmentation
    aug_fn = get_aug(aug if split == "train" or aug_test else "", dataset.graph.x)

    return DataLoader(
        dataset, batch_sampler=sampler, num_workers=num_workers,
        collate_fn=Collator(label_embeddings, aug=aug_fn)
    )
```

Then register and run:

```bash
# Add to data_loader_wrapper.py and trainer.py as shown above

# Run inference
python experiments/run_single_experiment.py \
  --dataset simple \
  --root ./data/simple \
  -way 3 -shot 3 -qry 5 \
  -pretrained checkpoint.ckpt \
  --eval_only True
```

## Next Steps

1. **Start simple**: Use the synthetic example above to verify your setup
2. **Load your data**: Replace the synthetic graph with your actual data
3. **Add text features**: If you have node/edge descriptions, use BERT
4. **Test inference**: Run with a pretrained checkpoint
5. **Fine-tune**: If needed, train on a small subset of your data

For more examples, see:
- `data/arxiv.py` - Citation network example
- `data/kg.py` - Knowledge graph example
- `data/mag240m.py` - Large-scale graph example
