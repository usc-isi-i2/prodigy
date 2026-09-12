"""CPU-only graph load/CSR cache benchmark; never modifies source artifacts."""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
import torch
from torch_sparse import SparseTensor
from experiments.sampler import preprocess

p = argparse.ArgumentParser()
p.add_argument('--graph', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--threads', type=int, default=4)
p.add_argument('--repeats', type=int, default=3)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
torch.set_num_threads(a.threads)
report = dict(graph=str(a.graph), torch=torch.__version__, threads=a.threads,
              note='OS caches not flushed; graph load plus default-edge CSR only, not full dataset startup', runs=[])
cache = a.output / 'csr.pt'

def timed(fn):
    start = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - start

for repeat in range(a.repeats):
    for mode in ('baseline', 'cached'):
        raw, load_s = timed(lambda: torch.load(a.graph, map_location='cpu'))
        n = raw['x'].shape[0]
        if mode == 'baseline':
            adj, adjacency_s = timed(lambda: preprocess(raw['edge_index'], n))
            if repeat == 0:
                arrays = adj.csr()
                _, save_s = timed(lambda: torch.save(dict(rowptr=arrays[0], col=arrays[1], value=arrays[2], n=n), cache))
                restored = torch.load(cache)
                assert all(torch.equal(arrays[i], restored[key]) for i, key in enumerate(('rowptr','col','value')))
                report.update(cache_bytes=cache.stat().st_size, save_s=save_s, exact_csr_roundtrip=True,
                              nodes=n, edges=raw['edge_index'].shape[1])
                del arrays, restored
        else:
            def restore():
                v = torch.load(cache, map_location='cpu')
                return SparseTensor(rowptr=v['rowptr'], col=v['col'], value=v['value'], sparse_sizes=(v['n'], v['n']), is_sorted=True, trust_data=True)
            adj, adjacency_s = timed(restore)
        _, sharing_s = timed(adj.share_memory_)
        row = dict(repeat=repeat, mode=mode, load_s=load_s, adjacency_s=adjacency_s,
                   sharing_s=sharing_s, total_s=load_s+adjacency_s+sharing_s)
        report['runs'].append(row)
        print(json.dumps(row), flush=True)
        (a.output/'timings.json').write_text(json.dumps(report, indent=2))
        del raw, adj
        gc.collect()
