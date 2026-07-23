"""Validate embed_nodes' encoder walk with stub layers (no compiled PyG exts).

Checks the things most likely to be wrong in a hand-rolled reimplementation of
SingleLayerGeneralGNN.forward:
  * layer dispatch by marker class, and that MetagraphLayer is skipped
  * forward() argument signatures per layer type
  * supernode indexing across a Batch (supernode is NOT auto-offset by PyG, so
    it needs + ptr[:-1]; edge_index_supernode IS auto-offset because its name
    contains "index")
  * output row order matches the requested node_ids
"""
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from models.layer_classes import BackgroundGNNLayer, SupernodeAggrLayer, MetagraphLayer
from scripts.eval.pair_link_eval import embed_nodes

ok = True
def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")

print("embed_nodes walk test (stub layers)")

meta_calls = []

class StubBG(torch.nn.Module, BackgroundGNNLayer):
    def forward(self, x_orig, x, edge_index, edge_attr, edge_index_supernode, ptr, batch):
        # signature must match general_gnn.py's BackgroundGNNLayer call site
        assert edge_index.dtype == torch.long
        return x * 2.0

class StubAggr(torch.nn.Module, SupernodeAggrLayer):
    def forward(self, x, supernode_edge_index, supernode_idx, graph_batch):
        # mimic mean-aggregation over the single (local-0 -> supernode) edge
        return x[supernode_edge_index[0]]

class StubMeta(torch.nn.Module, MetagraphLayer):
    def forward(self, *a, **k):
        meta_calls.append(1)
        raise AssertionError("MetagraphLayer must be skipped by embed_nodes")

class StubModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_list = torch.nn.ModuleList([StubBG(), StubAggr(), StubMeta()])
        self.initial_input_mlp = torch.nn.Identity()
        self.final_input_mlp = torch.nn.Identity()
        self.txt_dropout = None
        self.params = {"skip_path": False}

FEAT = 3
N = 12
feats = torch.arange(N * FEAT, dtype=torch.float).reshape(N, FEAT)

class StubDS:
    """One subgraph per node: [center, one neighbour] + appended supernode."""
    def __getitem__(self, i):
        nbr = (i + 1) % N
        x = torch.stack([feats[i], feats[nbr], torch.zeros(FEAT)])
        d = Data(x=x, edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
                 num_nodes=3)
        d.supernode = torch.tensor([2])
        d.edge_index_supernode = torch.tensor([[0], [2]], dtype=torch.long)
        return d

model = StubModel()
node_ids = [3, 7, 1, 9, 0]
emb = embed_nodes(model, StubDS(), node_ids, device="cpu", batch_size=2)

check("metagraph layer was skipped", len(meta_calls) == 0)
check("output shape is (len(node_ids), feat)", emb.shape == (len(node_ids), FEAT),
      f"{emb.shape}")

expected = np.stack([(feats[i] * 2.0).numpy() for i in node_ids])
check("row order and values match the requested node_ids",
      np.allclose(emb, expected), f"max_err={np.abs(emb - expected).max():.2e}")

# batch_size=1 must give identical results to batching (catches offset bugs that
# only appear when several subgraphs share a batch)
emb1 = embed_nodes(model, StubDS(), node_ids, device="cpu", batch_size=1)
check("batching is consistent with batch_size=1", np.allclose(emb, emb1),
      f"max_err={np.abs(emb - emb1).max():.2e}")

# missing supernode aggregation must raise, not silently return garbage
class NoAggrModel(StubModel):
    def __init__(self):
        super().__init__()
        self.layer_list = torch.nn.ModuleList([StubBG()])
try:
    embed_nodes(NoAggrModel(), StubDS(), [0, 1], device="cpu")
    check("missing supernode layer is rejected", False, "no error raised")
except RuntimeError as e:
    check("missing supernode layer is rejected", "pooled embedding" in str(e))

print(f"\n{'ALL CHECKS PASSED' if ok else 'FAILURES PRESENT'}")
sys.exit(0 if ok else 1)
