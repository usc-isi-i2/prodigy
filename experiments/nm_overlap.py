"""Training-only support relation masks for overlap-aware neighbor matching."""
import torch


def overlap_aware_support_keep_mask(
    rowptr,
    col,
    sample_centers,
    task_ids,
    task_anchors,
    metagraph_edge_index,
    metagraph_edge_attr,
):
    """Keep every edge except false-negative support-to-label messages.

    ``rowptr``/``col`` are the bidirectional training-view CSR used to select NM
    members. Query edges, positive support edges, self loops added inside the
    metagraph, and decoder candidates are unaffected.
    """
    if metagraph_edge_index.ndim != 2 or metagraph_edge_index.shape[0] != 2:
        raise ValueError("metagraph_edge_index must have shape [2, edges]")
    if metagraph_edge_attr.shape != (metagraph_edge_index.shape[1], 2):
        raise ValueError("expected two metagraph edge attributes")
    sample_centers = torch.as_tensor(sample_centers, dtype=torch.long).reshape(-1).cpu()
    task_ids = torch.as_tensor(task_ids, dtype=torch.long).reshape(-1).cpu()
    task_anchors = torch.as_tensor(task_anchors, dtype=torch.long).cpu()
    edges = metagraph_edge_index.cpu()
    attrs = metagraph_edge_attr.cpu()
    if len(sample_centers) != len(task_ids) or task_anchors.ndim != 2:
        raise ValueError("sample/task metadata dimensions differ")
    if len(sample_centers) == 0 or task_anchors.numel() == 0:
        raise ValueError("empty NM episode")

    source = edges[0]
    destination = edges[1]
    n_samples = len(sample_centers)
    n_way = task_anchors.shape[1]
    candidate = destination - n_samples - task_ids[source] * n_way
    if candidate.min() < 0 or candidate.max() >= n_way:
        raise ValueError("metagraph label nodes do not match task metadata")
    negative_support = source.lt(n_samples) & attrs[:, 0].eq(0) & attrs[:, 1].lt(0)
    keep = torch.ones(edges.shape[1], dtype=torch.bool)

    # CSR columns are sorted. Check all candidate anchors for each support row
    # together instead of materializing a global edge set.
    for sample in torch.unique(source[negative_support]).tolist():
        selected = negative_support & source.eq(sample)
        candidates = candidate[selected]
        anchors = task_anchors[task_ids[sample], candidates]
        node = int(sample_centers[sample])
        neighbors = col[rowptr[node]:rowptr[node + 1]]
        if len(neighbors) == 0:
            continue
        positions = torch.searchsorted(neighbors, anchors)
        valid = positions.lt(len(neighbors))
        hits = torch.zeros_like(valid)
        hits[valid] = neighbors[positions[valid]].eq(anchors[valid])
        selected_indices = torch.where(selected)[0]
        keep[selected_indices[hits]] = False

    if (~keep & ~negative_support).any():
        raise AssertionError("overlap mask removed a nonnegative or query edge")
    return keep
