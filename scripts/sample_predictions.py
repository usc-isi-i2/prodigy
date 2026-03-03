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
from tqdm import trange

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.midterm import get_midterm_dataset, get_midterm_dataloader
from models.general_gnn import SingleLayerGeneralGNN
from experiments.layers import get_module_list

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
emb_dim = 256
params = {
    'input_dim': args.input_dim, 'emb_dim': emb_dim, 'gnn_type': 'sage',
    'n_layer': 1, 'meta_n_layer': 1, 'second_gnn': 'Atten',
    'attention_mask_scheme': 'causal', 'skip_path': False,
    'has_final_back': False, 'layers': 'S,U,M',
    'ignore_label_embeddings': True, 'zero_label_embeddings': False,
    'not_freeze_learned_label_embedding': False, 'linear_probe': False,
    'no_bn_metagraph': False, 'no_bn_encoder': False,
    'dropout': 0, 'reset_after_layer': None, 'meta_gnn_pos_only': False,
    'text_features_dropout': 0, 'zero_shot': False,
}
initial_label_mlp = torch.nn.Linear(768, emb_dim)
layer_list = get_module_list(
    params['layers'], emb_dim, edge_attr_dim=None,
    input_dim=params['input_dim'], dropout=params['dropout'],
    reset_after_layer=params['reset_after_layer'],
    attention_mask_scheme=params['attention_mask_scheme'],
    has_final_back=params['has_final_back'],
    msg_pos_only=params['meta_gnn_pos_only'],
    batch_norm_metagraph=True, batch_norm_encoder=True, gnn_use_relu=False,
)
layer_list = torch.nn.ModuleList(layer_list)
txt_dropout = torch.nn.Dropout(0)
model = SingleLayerGeneralGNN(layer_list=layer_list, initial_label_mlp=initial_label_mlp, params=params, text_dropout=txt_dropout)
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
    for episode_idx, batch in zip(trange(args.n_episodes, desc='episodes'), dataloader):
        batch = [b.to(device) for b in batch]

        # --- labels / split (needed before model call) ---
        labels_onehot = batch[2].detach().cpu()
        num_labels = labels_onehot.shape[1]
        gt_label_idx = torch.argmax(labels_onehot, dim=1).long()
        meta_mask = batch[5].detach().cpu().view(-1, num_labels)
        is_query = meta_mask[:, 0].bool()

        # --- few-shot logistic regression on raw node features ---
        graph = batch[0]
        supernode_idx = (graph.supernode + graph.ptr[:-1]).long()
        raw_feats = graph.x[supernode_idx].cpu().numpy()
        support_mask = (~is_query).numpy()
        query_mask = is_query.numpy()
        support_labels_np = gt_label_idx[support_mask].numpy()
        classes = sorted(set(support_labels_np))
        centroids = np.stack([raw_feats[support_mask][support_labels_np == c].mean(0) for c in classes])
        dists = np.linalg.norm(raw_feats[query_mask][:, None] - centroids[None, :], axis=2)
        lr_preds = np.array([classes[i] for i in dists.argmin(axis=1)])

        # --- GNN model ---
        yt, yp, graph_out = model(*batch)
        yp_cpu = yp.detach().cpu()
        pred_idx = torch.argmax(yp_cpu, dim=1).long()

        # Get center node indices if available
        center_nodes = None
        if hasattr(graph_out, 'center_node_idx'):
            center_nodes = graph_out.center_node_idx.detach().cpu().flatten().tolist()

        q = 0  # index into yp_cpu / pred_idx / lr_preds (query nodes only)
        for i in range(len(gt_label_idx)):
            if not is_query[i]:
                continue  # skip support nodes
            node_idx = center_nodes[i] if center_nodes else None
            true_label = int(gt_label_idx[i].item())
            pred_label = int(pred_idx[q].item())
            lr_label = int(lr_preds[q])
            uid = int(user_ids[node_idx]) if (user_ids is not None and node_idx is not None) else node_idx
            true_state = label_names[true_label] if true_label < len(label_names) else true_label
            pred_state = label_names[pred_label] if pred_label < len(label_names) else pred_label
            lr_state = label_names[lr_label] if lr_label < len(label_names) else lr_label
            rows.append({
                'episode': episode_idx,
                'node_idx': node_idx,
                'user_id': uid,
                'true_state': true_state,
                'pred_state': pred_state,
                'correct': true_state == pred_state,
                'confidence': float(yp_cpu[q].max().item()),
                'lr_pred_state': lr_state,
                'lr_correct': true_state == lr_state,
            })
            q += 1

df = pd.DataFrame(rows)
df.to_csv(args.output, index=False)
print(f"\nSaved {len(df)} predictions to {args.output}")
print(f"GNN accuracy:      {df['correct'].mean():.4f}")
print(f"Nearest-centroid:  {df['lr_correct'].mean():.4f}")
print(f"Random baseline:   {1/args.n_way:.4f}")
print(f"\nSample predictions:")
print(df[['episode','true_state','pred_state','correct','lr_pred_state','lr_correct']].head(10).to_string(index=False))
