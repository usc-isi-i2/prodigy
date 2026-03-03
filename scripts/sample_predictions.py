"""
Run a few classification episodes and save predictions to CSV.
Usage:
    python scripts/sample_predictions.py \
        --root midterm/graph_co_retweet \
        --checkpoint state/finetune_classification_02_03_2026_13_03_24/checkpoint/state_dict_2000.ckpt \
        --n_episodes 20 \
        --output predictions.csv
"""
import argparse
import sys
import os
import torch
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.params import get_params
from data.midterm import get_midterm_dataset, get_midterm_dataloader
from models.get_model import get_model

parser = argparse.ArgumentParser()
parser.add_argument('--root', required=True)
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--input_dim', type=int, default=98)
parser.add_argument('--n_episodes', type=int, default=20)
parser.add_argument('--n_way', type=int, default=3)
parser.add_argument('--n_shot', type=int, default=3)
parser.add_argument('--n_query', type=int, default=10)
parser.add_argument('--output', default='predictions.csv')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load graph
print("Loading dataset...")
dataset = get_midterm_dataset(root=args.root)
graph = dataset.graph
label_names = graph.label_names
user_ids = graph.user_ids if hasattr(graph, 'user_ids') else None

# Load raw graph for user_ids
raw = torch.load(os.path.join(args.root, 'graph_data.pt'), map_location='cpu')
user_ids = raw.get('user_ids', None)

# Build dataloader
dataloader = get_midterm_dataloader(
    dataset=dataset,
    split='test',
    node_split='test',
    batch_size=1,
    n_way=args.n_way,
    n_shot=args.n_shot,
    n_query=args.n_query,
    batch_count=args.n_episodes,
    root=args.root,
    bert=None,
    num_workers=0,
    aug='',
    aug_test=False,
    split_labels=False,
    train_cap=None,
    linear_probe=False,
    task_name='classification',
)

# Load model
print("Loading model...")
params = {
    'input_dim': args.input_dim, 'emb_dim': 256, 'gnn_type': 'sage',
    'n_layer': 1, 'meta_n_layer': 1, 'second_gnn': 'Atten',
    'attention_mask_scheme': 'causal', 'skip_path': False,
    'has_final_back': False, 'layers': 'S,U,M',
    'ignore_label_embeddings': True, 'zero_label_embeddings': False,
    'not_freeze_learned_label_embedding': False, 'linear_probe': False,
    'no_bn_metagraph': False, 'no_bn_encoder': False,
    'dropout': 0, 'device': device,
}
model = get_model(params)
model = model.to(device)

ckpt = torch.load(args.checkpoint, map_location=device)
for key, module in [('model', model)]:
    if key in ckpt:
        module.load_state_dict(ckpt[key], strict=False)
print("Model loaded.")

# Run episodes and collect predictions
rows = []
model.eval()
with torch.no_grad():
    for episode_idx, batch in enumerate(dataloader):
        batch = [b.to(device) for b in batch]
        yt, yp, graph_out = model(*batch)

        yt_cpu = yt.detach().cpu()
        yp_cpu = yp.detach().cpu()

        # Get center node indices if available
        center_nodes = None
        if hasattr(graph_out, 'center_node_idx'):
            center_nodes = graph_out.center_node_idx.detach().cpu().flatten().tolist()

        # Get label assignments (which n_way classes were sampled)
        labels_onehot = batch[2].detach().cpu()
        num_labels = labels_onehot.shape[1]
        gt_label_idx = torch.argmax(labels_onehot, dim=1).long()
        meta_mask = batch[5].detach().cpu().view(-1, num_labels)
        is_query = meta_mask[:, 0].bool()

        pred_idx = torch.argmax(yp_cpu, dim=1).long()

        for i in range(len(gt_label_idx)):
            if not is_query[i]:
                continue  # skip support nodes
            node_idx = center_nodes[i] if center_nodes else None
            true_label = int(gt_label_idx[i].item())
            pred_label = int(pred_idx[i].item())
            uid = int(user_ids[node_idx]) if (user_ids is not None and node_idx is not None) else node_idx
            true_state = label_names[true_label] if true_label < len(label_names) else true_label
            pred_state = label_names[pred_label] if pred_label < len(label_names) else pred_label
            rows.append({
                'episode': episode_idx,
                'node_idx': node_idx,
                'user_id': uid,
                'true_state': true_state,
                'pred_state': pred_state,
                'correct': true_state == pred_state,
                'confidence': float(yp_cpu[i].max().item()),
            })

df = pd.DataFrame(rows)
df.to_csv(args.output, index=False)
print(f"\nSaved {len(df)} predictions to {args.output}")
print(f"Overall accuracy: {df['correct'].mean():.4f}")
print(f"\nSample predictions:")
print(df.head(10).to_string(index=False))
