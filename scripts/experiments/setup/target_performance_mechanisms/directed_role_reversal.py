"""Reverse background relations without changing sampled membership or task roles."""
import torch
from .replay import clone_batch
from .role_context import query_mask

CONDITIONS = ('intact', 'support', 'query', 'both')


def reverse_roles(batch, condition):
    if condition not in CONDITIONS:
        raise ValueError('Unknown directed-role condition')
    out = clone_batch(batch)
    g = out[0]
    edge = g.edge_index
    if not torch.equal(g.batch[edge[0]], g.batch[edge[1]]):
        raise ValueError('Background edges cross subgraphs')
    if torch.any(g.global_node_ids[edge] < 0):
        raise ValueError('Background edges touch synthetic nodes')
    q = query_mask(batch)
    selected = q if condition=='query' else ~q if condition=='support' else torch.full_like(q,condition=='both')
    mask = selected[g.batch[edge[0]]]
    old = edge.clone()
    g.edge_index[:,mask] = old.flip(0)[:,mask]
    # Every edge slot keeps its unordered endpoints, attributes and multiplicity.
    torch.testing.assert_close(g.edge_index.sort(dim=0).values,old.sort(dim=0).values,rtol=0,atol=0)
    n = len(g.x)
    node_selected = selected[g.batch]
    for axis in (0,1):
        before = torch.bincount(old[axis],minlength=n)
        reverse = torch.bincount(old[1-axis],minlength=n)
        after = torch.bincount(g.edge_index[axis],minlength=n)
        torch.testing.assert_close(after,torch.where(node_selected,reverse,before),rtol=0,atol=0)
    return out, dict(condition=condition,edges=int(edge.shape[1]),
                     reversed_slots=int(mask.sum()),nonself_reversed_slots=int((mask & (old[0]!=old[1])).sum()),
                     degree_exchange_exact=True,unordered_endpoints_exact=True)
