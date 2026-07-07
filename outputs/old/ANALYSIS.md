# PRODIGY - Codebase Analysis

## Overview

PRODIGY (PRetraining framework enabling in-context learning Over Diverse tasks on unsen Graphs via prompting) is a research implementation for enabling in-context learning over graphs. The project implements a pretraining framework that allows graph neural networks to adapt to diverse downstream tasks on unseen graphs without parameter optimization, using few-shot prompting techniques.

**Paper**: [PRODIGY: Enabling In-context Learning Over Graphs](https://arxiv.org/abs/2305.12600) (SPIGM @ ICML 2023)

**Authors**: Qian Huang, Hongyu Ren, Peng Chen, Gregor Kržmanc, Daniel Zeng, Percy Liang, Jure Leskovec

---

## Project Structure

### Core Components

```
prodigy/
├── data/                           # Data loading and preprocessing
│   ├── arxiv.py                    # arXiv dataset handling
│   ├── mag240m.py                  # MAG240M dataset handling
│   ├── kg.py                       # Knowledge graph dataset handling
│   ├── dataloader.py               # Custom dataloaders for few-shot tasks
│   ├── dataset.py                  # Base dataset classes
│   ├── augment.py                  # Data augmentation utilities
│   └── load_kg_dataset.py          # KG dataset loading utilities
├── models/                         # Model architectures
│   ├── metaGNN.py                  # Core meta-learning GNN layers
│   ├── general_gnn.py              # General GNN architecture
│   ├── multilayer_gnn.py           # Multi-layer GNN implementations
│   ├── gnn_with_edge_attr.py       # GNN with edge attributes
│   ├── layer_classes.py            # Custom layer implementations
│   ├── get_model.py                # Model factory and utilities
│   ├── sentence_embedding.py       # Sentence embedding utilities
│   └── model_eval_utils.py         # Evaluation utilities
├── experiments/                    # Training and evaluation
│   ├── run_single_experiment.py    # Main entry point for experiments
│   ├── trainer.py                  # Training loop implementation
│   ├── params.py                   # Hyperparameter definitions
│   ├── layers.py                   # Layer configuration utilities
│   └── sampler.py                  # Custom sampling strategies
└── kg_commands.py                  # Command templates for KG datasets
```

---

## Key Features

### 1. **Meta-Learning Architecture**

The project implements several sophisticated GNN architectures designed for meta-learning:

- **MetaGNN**: Custom attention-based message passing for bipartite graphs with support/query sets
- **MetaTransformer**: Transformer-based architecture for sequential processing of graph prompts
- **MetaGATConv**: Graph Attention Network variant for metagraph processing
- **Multi-layer GNN**: Flexible architecture supporting various layer types (SAGE, GIN, GAT, etc.)

### 2. **Few-Shot Learning Tasks**

Multiple pretraining tasks are supported:

- **Classification (cls_nm_sb)**: Node/edge classification with neighbor matching
- **Neighbor Matching (neighbor_matching)**: Learning to match graph structure patterns
- **Multiway Classification**: Multi-class classification with varying number of classes
- **Same Graph Contrastive**: Contrastive learning on graph augmentations

### 3. **Data Augmentation**

Sophisticated augmentation strategies for graphs:

- **Node Dropping (ND)**: Randomly drop nodes with specified probability
- **Node Zeroing (NZ)**: Zero out node features
- Applicable during both training and testing

### 4. **Knowledge Graph Support**

Specialized support for KG datasets:

- Wiki, FB15K-237, NELL, ConceptNet
- Relation-based few-shot learning
- Custom edge attribute handling
- Sentence transformer embeddings for entities and relations

### 5. **Flexible Dataset Support**

- **arXiv**: Citation network with category classification
- **MAG240M**: Large-scale academic graph
- **Knowledge Graphs**: Wiki, FB15K-237, NELL, ConceptNet
- Automatic preprocessing and caching

---

## Technical Implementation Details

### Model Architecture

The main model (`SingleLayerGeneralGNN` in `general_gnn.py`) follows a modular design:

1. **Input Processing Layer**: Encodes node features (uses BERT/RoBERTa embeddings for text)
2. **Graph Encoder Layers**: Multiple GNN layers (SAGE, GIN, GAT)
3. **Up-sampling Layer**: Prepares features for metagraph
4. **Metagraph Layer**: Processes support/query bipartite graph structure
5. **Classification Head**: Produces predictions for query nodes

### Meta-Learning Components

**MetaGNN Layer** (`metaGNN.py:75-143`):
- Multi-head attention mechanism
- Edge attribute incorporation
- Bipartite graph message passing
- Separates support set and query set processing

**Attention Mechanism**:
```python
# Computes: alpha = softmax(MLP([k, q, edge_attr]))
# Message: m = alpha * v
```

### Training Pipeline

**Trainer** (`trainer.py:21-514`):
1. Initialize model and datasets
2. Create few-shot task samplers
3. Training loop with:
   - Forward pass on support/query sets
   - Loss computation (CrossEntropy or BCEWithLogits)
   - Auxiliary loss for masked node attribute prediction
   - Validation on unseen tasks
   - Checkpoint saving
4. Evaluation with metrics:
   - Accuracy
   - Hits@1, Hits@5, Hits@10
   - Mean Reciprocal Rank (MRR)

### Few-Shot Task Sampling

**Key Classes** (`dataloader.py`):

- **IsomorphismTask**: Sample graphs by ID
- **MultiTaskSplitWay**: Split tasks evenly or randomly across multiple datasets
- **Custom Samplers**: Support varying n-way, k-shot configurations during training

### Data Flow

```
Raw Dataset → Preprocessing → Cache
    ↓
Few-Shot Task Sampler
    ↓
Support Set + Query Set → Bipartite Graph
    ↓
Graph Encoder → Metagraph Processor → Classifier
    ↓
Predictions & Loss
```

---

## Configuration System

The project uses an extensive command-line argument system (`params.py`):

**Key Parameters**:

- **Dataset**: `--dataset` (arxiv, mag240m, Wiki, FB15K-237, etc.)
- **Architecture**: `--layers` (e.g., "S2,U,M" = 2 SAGE layers + Up + Metagraph)
- **Few-shot**: `--n_way`, `--n_shots`, `--n_query`
- **Training**: `--lr`, `--epochs`, `--batch_size`
- **Tasks**: `--task_name` (classification, neighbor_matching, etc.)
- **Augmentation**: `--aug` (e.g., "ND0.5,NZ0.5")

**Layer Notation** (`layers.py`):
- `S`: GraphSAGE layer
- `U`: Upsample layer
- `M`: Metagraph layer
- `A`: Average pooling
- Numbers indicate layer count (e.g., `S2` = 2 SAGE layers)

---

## Pretraining & Evaluation Workflow

### 1. Pretraining on MAG240M

```bash
python experiments/run_single_experiment.py \
  --dataset mag240m \
  --root <DATA_ROOT> \
  --layers S2,U,M \
  -way 30 -shot 3 -qry 4 \
  -task cls_nm_sb \
  -aug ND0.5,NZ0.5
```

Key aspects:
- 30-way classification (diverse tasks)
- 3 support examples per class
- 4 query examples per class
- Node dropping and feature zeroing augmentation

### 2. Evaluation on arXiv (Zero-shot Transfer)

```bash
python experiments/run_single_experiment.py \
  --dataset arxiv \
  --root <DATA_ROOT> \
  -pretrained <PATH_TO_CHECKPOINT> \
  --eval_only True \
  -way 3 -shot 3 -qry 3
```

Evaluates pretrained model on new graph without fine-tuning.

---

## Advanced Features

### 1. **Auxiliary Tasks**

**Masked Attribute Regression** (`trainer.py:295-305`):
- Predicts masked node features
- Adds regularization term to loss
- Controlled by `--attr_regression_weight` parameter

### 2. **Prompt Graph Construction**

The system constructs a bipartite "prompt graph":
- Left side: Query nodes (to classify)
- Right side: Label nodes (representing classes)
- Edges: Connect queries to potential labels
- Edge attributes: Indicate positive/negative examples

### 3. **Attention Masking Schemes**

For transformer-based metagraph layers:
- **Causal**: Standard autoregressive masking
- **Special**: Query tokens don't attend to each other
- **Mask**: Full masking with custom patterns

### 4. **Multi-task Learning**

Supports training on multiple datasets simultaneously:
- Even split: Equal tasks from each dataset
- Random split: Random distribution of tasks

---

## Logging & Monitoring

**Weights & Biases Integration**:
- Automatic experiment tracking
- Model checkpoint versioning
- Metric logging (loss, accuracy, MRR, Hits@k)
- Code versioning

**Checkpointing**:
- Periodic saves during training
- Best model selection based on validation accuracy
- Resume from checkpoint support

---

## Dependencies

**Core Libraries**:
- PyTorch 2.0.1
- PyTorch Geometric 2.3.1
- Transformers 4.29.2 (for BERT embeddings)
- Sentence-Transformers 2.2.2

**Specialized**:
- OGB (Open Graph Benchmark)
- torch_scatter, torch_sparse (efficient graph operations)
- wandb (experiment tracking)

---

## Research Contributions

1. **In-Context Learning for Graphs**: Enables GPT-like prompting for GNNs
2. **Task-Agnostic Pretraining**: Single pretraining works across diverse tasks
3. **Zero-Shot Graph Transfer**: Adapt to new graphs without retraining
4. **Scalable Meta-Learning**: Handles large-scale graphs (240M nodes for MAG240M)

---

## Implementation Highlights

### Efficient Design Choices

1. **Caching**: Preprocessed data cached to disk (LMDB format)
2. **Batch Processing**: Custom collate functions for variable graph sizes
3. **Memory Management**: Gradient checkpointing, mixed precision support potential
4. **Parallel Data Loading**: Multi-worker dataloaders

### Code Quality

- Modular architecture with clear separation of concerns
- Extensive parameterization for reproducibility
- Type hints and documentation in key areas
- Error handling and validation

---

## Usage Patterns

### Basic Experiment

```bash
# 1. Pretrain
python experiments/run_single_experiment.py \
  --dataset mag240m \
  --root ./data \
  --layers S2,U,M \
  -way 30 -shot 3 -qry 4 \
  --prefix MY_EXPERIMENT

# 2. Evaluate
python experiments/run_single_experiment.py \
  --dataset arxiv \
  --root ./data \
  -pretrained ./state/MY_EXPERIMENT_*/checkpoint/state_dict_*.ckpt \
  --eval_only True
```

### Custom Layer Configuration

```python
# In layers.py, define custom architecture:
"S2,U,M" → 2xSAGE + Upsample + Metagraph
"S2,UX,M2" → 2xSAGE + UpX + 2xMetagraph
"S2,UX,A" → 2xSAGE + UpX + Average
```

---

## Potential Extensions

Based on the codebase structure, potential research directions:

1. **New Architectures**: Add transformer encoders, graph transformers
2. **Additional Tasks**: Link prediction, graph classification, subgraph matching
3. **Improved Augmentation**: Learnable augmentation strategies
4. **Efficiency**: Quantization, distillation, pruning
5. **Datasets**: Extend to biological networks, social graphs, code graphs

---

## Known Limitations

1. **Memory Requirements**: Large graphs require significant GPU memory
2. **Preprocessing Time**: Initial dataset processing can be slow
3. **Hyperparameter Sensitivity**: Performance depends on careful tuning
4. **Limited Documentation**: Some advanced features lack detailed docs

---

## Citation

If using this codebase, cite:

```bibtex
@article{Huang2023PRODIGYEI,
  title={PRODIGY: Enabling In-context Learning Over Graphs},
  author={Qian Huang and Hongyu Ren and Peng Chen and Gregor Kržmanc and Daniel Zeng and Percy Liang and Jure Leskovec},
  journal={ArXiv},
  year={2023},
  volume={abs/2305.12600}
}
```

---

## Summary

PRODIGY is a well-engineered research codebase implementing meta-learning for graph neural networks. It demonstrates sophisticated techniques including:

- Few-shot learning on graphs
- Meta-learning with support/query paradigm
- Flexible GNN architectures
- Large-scale graph pretraining
- Zero-shot transfer to new graphs

The code is production-quality research software with good modularity, extensive configuration options, and support for multiple datasets and tasks. It represents a significant contribution to the field of graph representation learning and meta-learning.
