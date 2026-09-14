from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .config import load_config
from .evaluate_lattice import unwrap
from .precomputed_context import sampled_neighbor_means, tensor_sha256
from .train import configure_sampling_backend


def timed_direct_scans(x, neighborhood, batch_size, repeats):
    started = time.monotonic(); checksum = 0.0
    for _ in range(repeats):
        for start in range(0, len(neighborhood), batch_size):
            stop = min(len(neighborhood), start + batch_size)
            view = torch.cat((x[start:stop], neighborhood[start:stop]), dim=1)
            checksum += float(view[:, :1].sum())
    return time.monotonic() - started, checksum


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); p.add_argument("--graphs",nargs="+",required=True); p.add_argument("--output",required=True); p.add_argument("--limit-nodes",type=int); p.add_argument("--repeats",type=int,default=10); p.add_argument("--seed",type=int,default=0); a=p.parse_args()
    config=load_config(a.config); protocol=config["protocol"]; configure_sampling_backend(protocol); rows=[]
    for name in a.graphs:
        graph=unwrap(config["graphs"][name]["path"]); count=min(len(graph.x),a.limit_nodes or len(graph.x)); x=graph.x[:count].float()
        started=time.monotonic(); first=sampled_neighbor_means(graph.x,graph.edge_index,int(protocol["fanout"]),a.seed,int(protocol["eval_batch_size"]),limit_nodes=count); materialize=time.monotonic()-started
        started=time.monotonic(); second=sampled_neighbor_means(graph.x,graph.edge_index,int(protocol["fanout"]),a.seed,int(protocol["eval_batch_size"]),limit_nodes=count); replay=time.monotonic()-started
        direct,checksum=timed_direct_scans(x,first,int(protocol["eval_batch_size"]),a.repeats)
        rows.append({"graph":name,"nodes":count,"fanout":int(protocol["fanout"]),"seed":a.seed,"materialize_seconds":materialize,"deterministic_replay_seconds":replay,"direct_scan_seconds":direct,"direct_repeats":a.repeats,"estimated_repeated_sampling_seconds":replay*a.repeats,"estimated_speedup_after_cache":(replay*a.repeats)/direct,"max_replay_difference":float((first-second).abs().max()),"neighbor_mean_sha256":tensor_sha256(first),"checksum":checksum,"cache_bytes":first.numel()*first.element_size()})
        print(json.dumps(rows[-1]),flush=True)
    output=Path(a.output); output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps({"status":"complete","rows":rows},indent=2)+"\n")
    return 0

if __name__=="__main__": raise SystemExit(main())
