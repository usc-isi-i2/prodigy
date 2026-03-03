"""
Evaluate a trained model on temporal link prediction and save sample predictions to CSV.

Each episode: n_way center nodes, each with n_shot support neighbors + n_query query neighbors
(all drawn from future co-retweet edges). The GNN uses the history graph for message passing.
The model must classify each query node to its correct center node.

Usage:
    python scripts/sample_temporal_link_pred.py \
        --root midterm/graph_temporal \
        --checkpoint state/<run>/state_dict \
        --n_episodes 100 \
        --output temporal_preds.csv
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
parser.add_argument('--root',       required=True,  help='Path to graph_temporal dir')
parser.add_argument('--checkpoint', required=True,  help='Path to .ckpt or state_dict dir')
parser.add_argument('--input_dim',  type=int, default=98)
parser.add_argument('--n_episodes', type=int, default=100)
parser.add_argument('--n_way',      type=int, default=3)
parser.add_argument('--n_shot',     type=int, default=3)
parser.add_argument('--n_query',    type=int, default=10)
parser.add_argument('--output',     default='temporal_preds.csv')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── Load graph ────────────────────────────────────────────────────────────────
print("Loading dataset...")
dataset = get_midterm_dataset(root=args.root)
graph = dataset.graph

raw = torch.load(os.path.join(args.root, 'graph_data.pt'), map_location='cpu')
user_ids = raw.get('user_ids', None)  # array: node_idx → twitter user ID

# ── Build dataloader ──────────────────────────────────────────────────────────
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
    task_name='temporal_link_prediction',
)

# ── Load model ────────────────────────────────────────────────────────────────
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
model = SingleLayerGeneralGNN(
    layer_list=layer_list, initial_label_mlp=initial_label_mlp,
    params=params, text_dropout=txt_dropout,
)
model = model.to(device)

# Support both a raw state_dict dir and a .ckpt file
ckpt_path = args.checkpoint
if os.path.isdir(ckpt_path):
    # pick the latest .ckpt inside
    ckpts = sorted(f for f in os.listdir(ckpt_path) if f.endswith('.ckpt'))
    if not ckpts:
        raise FileNotFoundError(f"No .ckpt files found in {ckpt_path}")
    ckpt_path = os.path.join(ckpt_path, ckpts[-1])
    print(f"Using checkpoint: {ckpt_path}")

ckpt = torch.load(ckpt_path, map_location=device)
for key, module in [('model', model)]:
    if key in ckpt:
        module.load_state_dict(ckpt[key], strict=False)
print("Model loaded.")

# ── Run episodes ──────────────────────────────────────────────────────────────
rows = []
model.eval()
with torch.no_grad():
    for episode_idx, batch in zip(trange(args.n_episodes, desc='episodes'), dataloader):
        batch = [b.to(device) for b in batch]

        labels_onehot = batch[2].detach().cpu()
        num_labels = labels_onehot.shape[1]           # = n_way
        gt_label_idx = torch.argmax(labels_onehot, dim=1).long()
        meta_mask = batch[5].detach().cpu().view(-1, num_labels)
        is_query = meta_mask[:, 0].bool()

        g = batch[0]
        supernode_idx = (g.supernode + g.ptr[:-1]).long().cpu()

        yt, yp, graph_out = model(*batch)
        yp_cpu = yp.detach().cpu()
        pred_idx = torch.argmax(yp_cpu, dim=1).long()

        q = 0
        for i in range(len(gt_label_idx)):
            if not is_query[i]:
                continue
            node_idx = int(supernode_idx[i].item())
            uid = int(user_ids[node_idx]) if user_ids is not None else node_idx
            true_lbl = int(gt_label_idx[i].item())
            pred_lbl = int(pred_idx[q].item())
            rows.append({
                'episode':    episode_idx,
                'node_idx':   node_idx,
                'user_id':    uid,
                'true_label': true_lbl,   # which center node (0..n_way-1) it's a future neighbor of
                'pred_label': pred_lbl,
                'correct':    true_lbl == pred_lbl,
                'confidence': float(yp_cpu[q].max().item()),
            })
            q += 1

# ── Save & report ─────────────────────────────────────────────────────────────
df = pd.DataFrame(rows)
df.to_csv(args.output, index=False)

n_total   = len(df)
n_correct = df['correct'].sum()
acc       = df['correct'].mean()
random_baseline = 1.0 / args.n_way

print(f"\n{'='*50}")
print(f"Episodes:        {args.n_episodes}")
print(f"Query nodes:     {n_total}  ({n_total // args.n_episodes} per episode)")
print(f"GNN accuracy:    {acc:.4f}  ({n_correct}/{n_total} correct)")
print(f"Random baseline: {random_baseline:.4f}  (1/{args.n_way})")
print(f"Lift over random: {acc - random_baseline:+.4f}")
print(f"\nSaved {n_total} predictions to {args.output}")
print(f"\nSample predictions:")
print(df[['episode', 'user_id', 'true_label', 'pred_label', 'correct', 'confidence']].head(15).to_string(index=False))
