from typing import Optional, List, NamedTuple, Tuple, Union
import torch
import os
from tqdm import trange

from torch import Tensor
from torch_geometric.loader import NeighborSampler
from torch_sparse import SparseTensor, coalesce


def preprocess(edge_index, num_nodes=None, bidirectional=True):
    N = int(edge_index.max() + 1) if num_nodes is None else num_nodes
    edge_attr = torch.arange(edge_index.size(1))
    if bidirectional:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_attr = torch.cat([edge_attr, -1 - edge_attr], dim=0)
    whole_adj = SparseTensor.from_edge_index(edge_index, edge_attr, (N, N), is_sorted=False)

    rowptr, col, value = whole_adj.csr()  # convert to csr form
    whole_adj = SparseTensor(rowptr=rowptr, col=col, value=value, sparse_sizes=(N, N), is_sorted=True, trust_data=True)
    return whole_adj


def sample_k_hop_subgraph(
    node_idx: Union[int, List[int], Tensor],
    num_hops: int,
    whole_adj: Tensor,
    num_nodes: Optional[int] = None,
    bidirectional: bool = True,
    size: int = 100,
    limit: int = 2000,
    hop_sizes: Optional[List[int]] = None,
):
    '''
    input: similar to k_hop_subgraph (check https://pytorch-geometric.readthedocs.io/en/1.5.0/modules/utils.html?highlight=subgraph#torch_geometric.utils.k_hop_subgraph)
      key difference:
        (1) we need preprocess function and achieve the adj (of type SparseTensor)
        (2) argument `size` is added to control the number of nodes in the sampled subgraph
    output:
      n_id: the nodes involved in the subgraph
      edge_index: the edge_index in the subgraph
      edge_id: the id of edges in the subgraph
    '''
    if isinstance(node_idx, int):
        node_idx = torch.tensor([node_idx])
    elif isinstance(node_idx, list):
        node_idx = torch.tensor(node_idx)

    assert isinstance(whole_adj, SparseTensor)

    if hop_sizes is None:
        hop_sizes = [size] * num_hops
    if len(hop_sizes) != num_hops:
        raise ValueError(
            f"hop_sizes must have one entry per hop: num_hops={num_hops}, "
            f"hop_sizes={hop_sizes}"
        )
    if any(hop_size <= 0 for hop_size in hop_sizes):
        raise ValueError(f"hop_sizes must be positive, got {hop_sizes}")
    if limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}")

    adjs = []
    for hop_size in hop_sizes:
        adj, node_idx = whole_adj.sample_adj(node_idx, hop_size, replace=False)
        adjs.append(adj.coo())
        if node_idx.size(0) >= limit:
            break
    row = torch.cat([adj[0] for adj in adjs])
    col = torch.cat([adj[1] for adj in adjs])
    e_id = torch.cat([adj[2] for adj in adjs])

    if node_idx.size(0) > limit:
        node_idx = node_idx[:limit]
        mask = (row < limit).logical_and(col < limit)
        row = row[mask]
        col = col[mask]
        e_id = e_id[mask]

    mask = e_id < 0
    row[mask], col[mask] = col[mask], row[mask]
    e_id[mask] = -e_id[mask] - 1
    edge_index = torch.stack([row, col], dim=0)

    node_count = node_idx.size(0)
    edge_index, e_id = coalesce(edge_index, e_id, node_count, node_count, "min")

    return node_idx, edge_index, e_id


class NeighborSampler:
    def __init__(
        self,
        graph, # pyg graph
        num_hops: int,
        size: int = 100,
        limit: int = 2000,
        hop_sizes: Optional[List[int]] = None,
        walk_hops: Optional[int] = None,
    ):
        self.num_hops = num_hops
        self.size = size
        self.limit = limit
        self.hop_sizes = list(hop_sizes) if hop_sizes is not None else None
        self.walk_hops = num_hops if walk_hops is None else int(walk_hops)
        if self.walk_hops <= 0:
            raise ValueError(f"walk_hops must be positive, got {self.walk_hops}")
        self.whole_adj = preprocess(graph.edge_index, graph.num_nodes)
        self.whole_adj.share_memory_()

    def sample_node(self, node_idx):
        return sample_k_hop_subgraph(
            node_idx,
            num_hops=self.num_hops,
            whole_adj=self.whole_adj,
            size=self.size,
            limit=self.limit,
            hop_sizes=self.hop_sizes,
        )
    
    def sample_edge(
        self,
        node_idx: Tensor,
        direction: str,
    ):
        """
        direction: "in", "out", "inout"
        return sampled edges that contain node_idx
        order of edges will change!!
        """
        node_idx, edge_index, e_id =  sample_k_hop_subgraph(
            node_idx,
            num_hops=1,
            whole_adj=self.whole_adj,
            size=1,
            limit=3*len(node_idx),
        )
        rev = edge_index[0] > edge_index[1]
        e_id[rev] = -e_id[rev] - 1
        return e_id

    def random_walk(
        self,
        node_idx: Tensor,
        direction: str,
    ):
        """
        direction: "in", "out", "inout"
        """
        rowptr, col, e_id = self.whole_adj.csr()
        for _ in range(self.walk_hops):
            row_start = rowptr[node_idx]
            row_end = rowptr[node_idx + 1]
            mask = row_start < row_end
            if not mask.any():
                return node_idx[:0]

            row_start = row_start[mask]
            row_end = row_end[mask]
            widths = row_end - row_start
            idx = (torch.rand(row_start.shape, device=row_start.device) * widths).long() + row_start
            next_node_idx = col[idx]

            if direction == "in":
                dir_mask = e_id[idx] < 0
            elif direction == "out":
                dir_mask = e_id[idx] >= 0
            else:
                dir_mask = torch.ones_like(idx, dtype=torch.bool)

            node_idx = next_node_idx[dir_mask]
        return node_idx


class NeighborSamplerCacheAdj(NeighborSampler):
    def __init__(
        self,
        cache_path,
        graph, # pyg graph
        num_hops: int,
        size: int = 100,
        limit: int = 2000,
        hop_sizes: Optional[List[int]] = None,
        walk_hops: Optional[int] = None,
    ):
        self.num_hops = num_hops
        self.size = size
        self.limit = limit
        self.hop_sizes = list(hop_sizes) if hop_sizes is not None else None
        self.walk_hops = num_hops if walk_hops is None else int(walk_hops)
        if self.walk_hops <= 0:
            raise ValueError(f"walk_hops must be positive, got {self.walk_hops}")
        if os.path.exists(cache_path):
            print(f"Loading adjacent matrix for neighbor sampling from {cache_path}")
            self.whole_adj = torch.load(cache_path)
            print(f"Loaded adjacent matrix for neighbor sampling from {cache_path}")
        else:
            print(f"Preprocessing adjacent matrix for neighbor sampling")
            self.whole_adj = preprocess(graph.edge_index, graph.num_nodes)
            print(f"Saving adjacent matrix for neighbor sampling to {cache_path}")
            torch.save(self.whole_adj, cache_path)
            print(f"Saved adjacent matrix for neighbor sampling to {cache_path}")
        self.whole_adj.share_memory_()


def parse_hop_sizes(value: Union[str, List[int], Tuple[int, ...], None], num_hops: int):
    """Parse an optional per-hop fanout specification without changing defaults."""
    if value is None or value == "":
        return None
    if isinstance(value, str):
        sizes = [int(part.strip()) for part in value.split(",") if part.strip()]
    else:
        sizes = [int(part) for part in value]
    if len(sizes) != num_hops:
        raise ValueError(
            f"neighbor_sampling_hop_sizes must contain {num_hops} values, got {sizes}"
        )
    if any(size <= 0 for size in sizes):
        raise ValueError(f"neighbor_sampling_hop_sizes must be positive, got {sizes}")
    return sizes


def sampler_kwargs_from_config(kwargs, num_hops: int):
    """Translate experiment kwargs into NeighborSampler options.

    Empty/default values reproduce the historical sampler exactly.
    """
    walk_hops = int(kwargs.get("neighbor_matching_walk_hops", 0) or 0)
    limit = int(kwargs.get("neighbor_sampling_node_limit", 2000))
    if limit <= 0:
        raise ValueError(f"neighbor_sampling_node_limit must be positive, got {limit}")
    return {
        "hop_sizes": parse_hop_sizes(
            kwargs.get("neighbor_sampling_hop_sizes", ""), num_hops
        ),
        "limit": limit,
        "walk_hops": walk_hops or None,
    }


if __name__ == '__main__':
    from ogb.lsc import MAG240MDataset, MAG240MEvaluator
    ROOT = '<DATA ROOT>'
    dataset = MAG240MDataset(ROOT)

    import numpy as np
    import torch
    from torch_geometric.data import Data

    edge_index = dataset.edge_index('paper', 'cites', 'paper')
    edge_index = torch.from_numpy(edge_index)
    print("Dataset loaded")
    graph = Data(edge_index=edge_index)
    print(graph)
    print(graph.num_nodes)

    nh = 2
    sampler = NeighborSamplerCacheAdj("test_adj.pt", graph, nh, relabel_nodes=True)

    num_nodes = graph.num_nodes # this is an expensive operation
    for i in trange(10000):
        a = np.random.choice(num_nodes)
        sampler.sample_node(i)
