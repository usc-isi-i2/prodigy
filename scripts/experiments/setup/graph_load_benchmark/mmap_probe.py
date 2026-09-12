"""Compare eager and mapped graph+CSR loading through actual neighborhood access."""
import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
import torch
from torch_sparse import SparseTensor
from experiments.sampler import sample_k_hop_subgraph

p = argparse.ArgumentParser()
p.add_argument('--graph', type=Path, required=True)
p.add_argument('--cache', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--repeats', type=int, default=3)
a = p.parse_args()
assert not a.output.exists()
torch.set_num_threads(4)
rows = []
reference = None
for repeat in range(a.repeats):
    # Alternate order to reduce systematic warm-cache/order bias.
    modes = ['eager', 'mmap', 'mmap_shared']
    if repeat % 2:
        modes.reverse()
    for mode in modes:
        print(f'Start {repeat} {mode}', flush=True)
        start = time.perf_counter()
        raw = torch.load(a.graph, map_location='cpu', weights_only=False, mmap=mode != 'eager')
        loaded = time.perf_counter()
        v = torch.load(a.cache, map_location='cpu', weights_only=True, mmap=mode != 'eager')
        adj = SparseTensor(rowptr=v['rowptr'], col=v['col'], value=v['value'], sparse_sizes=(v['n'],v['n']), is_sorted=True, trust_data=True)
        indexed = time.perf_counter()
        if mode != 'mmap':
            adj.share_memory_()
        shared = time.perf_counter()
        torch.manual_seed(42)
        centers = torch.randint(v['n'], (210,))
        digest = hashlib.sha256()
        for node in centers:
            nodes, edges, edge_ids = sample_k_hop_subgraph(node.reshape(1), num_hops=2, whole_adj=adj, hop_sizes=[9,9], limit=101)
            features = raw['x'][nodes]
            for tensor in (nodes, edges, edge_ids, features):
                digest.update(tensor.contiguous().numpy().tobytes())
        accessed = time.perf_counter()
        signature = digest.hexdigest()
        if reference is None:
            reference = signature
        assert signature == reference, 'Neighborhoods or features differ between modes'
        row = dict(repeat=repeat, mode=mode, graph_load_s=loaded-start,
                   csr_load_s=indexed-loaded, sharing_s=shared-indexed,
                   first_210_subgraphs_s=accessed-shared, total_s=accessed-start,
                   sample_sha256=signature)
        rows.append(row)
        print(json.dumps(row), flush=True)
        a.output.write_text(json.dumps(dict(torch=torch.__version__, graph=str(a.graph),
            note='Warm OS cache; fixed random centers, two-hop 9/9 sampling and features; excludes model and full dataset setup', runs=rows), indent=2))
        del raw, v, adj, nodes, edges, edge_ids, features
        gc.collect()
