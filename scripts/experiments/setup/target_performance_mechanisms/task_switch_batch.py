"""Native, fixed-input batches for the prespecified binary task switch.

Four episodes each contain twenty supports followed by twenty-four queries.
Sequence entries use the collator's identity permutation (no random state).
Ground-truth sequences are bookkeeping only, never a permitted model input
for the diagnostic: the runner must audit query-label blindness.
"""
import torch
from torch_geometric.data import Batch

N_EPISODES, N_POINTS, N_SUPPORT = 4, 44, 20


def _labels(labels, require_balance):
    if not isinstance(labels, torch.Tensor) or labels.shape != (176,):
        raise ValueError("rule_labels must be a CPU integer tensor of shape (176,)")
    if labels.device.type != "cpu" or labels.dtype not in (torch.int32, torch.int64):
        raise ValueError("rule_labels must contain CPU integers")
    if not bool(((labels == 0) | (labels == 1)).all()):
        raise ValueError("only binary rule labels are supported")
    labels = labels.long().clone()
    if require_balance:
        by_episode = labels.reshape(N_EPISODES, N_POINTS)
        if not bool((by_episode[:, :N_SUPPORT].sum(1) == 10).all()):
            raise ValueError("each episode needs ten supports per class")
        if not bool((by_episode[:, N_SUPPORT:].sum(1) == 12).all()):
            raise ValueError("each episode needs twelve queries per class")
    return labels


def _task_tensors(labels):
    points = torch.arange(176).reshape(N_EPISODES, N_POINTS)
    mask = torch.arange(N_POINTS).ge(N_SUPPORT).repeat(N_EPISODES)
    targets = 176 + 2 * torch.arange(N_EPISODES).repeat_interleave(N_POINTS)
    edge_index = torch.stack((points.flatten().repeat_interleave(2),
                             targets.repeat_interleave(2) + torch.arange(2).repeat(176)))
    edge_mask = mask.repeat_interleave(2)
    onehot = torch.nn.functional.one_hot(labels, 2).float()
    sign = (2 * onehot.flatten() - 1) * (~edge_mask)
    attrs = torch.stack((edge_mask.float(), sign), dim=1)
    correct_targets = (targets + labels).reshape(N_EPISODES, N_POINTS)

    def seq(point_indices, label_indices):
        return torch.stack((point_indices, label_indices), dim=-1).reshape(N_EPISODES, -1)

    input_seq = seq(points[:, :N_SUPPORT], correct_targets[:, :N_SUPPORT])
    query_seq = seq(points[:, N_SUPPORT:], torch.full((N_EPISODES, 24), 184))
    query_gt = seq(points[:, N_SUPPORT:], correct_targets[:, N_SUPPORT:])
    return onehot, edge_index, attrs, edge_mask, input_seq, query_seq, query_gt


def build_batch(graphs, rule_labels, label_table):
    """Return native nine-item list; never mutate supplied subgraphs or tensors."""
    labels = _labels(rule_labels, True)
    if len(graphs) != 176:
        raise ValueError("expected exactly four 44-point episodes")
    if (not isinstance(label_table, torch.Tensor) or label_table.shape != (2, 768)
            or label_table.device.type != "cpu" or not label_table.is_floating_point()
            or not bool(torch.isfinite(label_table).all())):
        raise ValueError("label_table must be finite floating CPU tensor, shape (2,768)")
    clean = []
    sources = []
    for graph in graphs:
        g = graph.clone()
        if any(value.device.type != "cpu" for _, value in g if isinstance(value, torch.Tensor)):
            raise ValueError("subgraphs must be on CPU")
        if getattr(g, "y", None) is not None:
            g.y = torch.zeros_like(g.y)
        clean.append(g)
    for episode in range(N_EPISODES):
        ids = set()
        for g in clean[episode * N_POINTS:(episode + 1) * N_POINTS]:
            if getattr(g, "graph_id", None) is None:
                continue
            center = 0
            if getattr(g, "global_node_ids", None) is not None and getattr(g, "center_node_idx", None) is not None:
                matches = torch.where(g.global_node_ids.reshape(-1) == int(g.center_node_idx))[0]
                if len(matches):
                    center = int(matches[0])
            ids.add(int(g.graph_id.reshape(-1)[center]))
        sources.append(next(iter(ids)) if len(ids) == 1 else -1)
    graph_batch = Batch.from_data_list(clean)
    graph_batch.task_id_per_sample = torch.arange(N_EPISODES).repeat_interleave(N_POINTS)
    graph_batch.task_id_per_query = torch.arange(N_EPISODES).repeat_interleave(24)
    graph_batch.source_id_per_task = torch.tensor(sources, dtype=torch.long)
    graph_batch.task_label_map = torch.arange(2).repeat(N_EPISODES, 1)
    return [graph_batch, label_table.clone().repeat(N_EPISODES, 1), *_task_tensors(labels)]


def relabel_batch(batch, rule_labels, *, require_balance=True):
    """Clone a diagnostic batch, changing only labels and derived bookkeeping.

    ``require_balance=False`` is reserved for the query-label leakage audit.
    It does not relax binary/integer/shape checks.
    """
    labels = _labels(rule_labels, require_balance)
    if len(batch) != 9 or batch[0].num_graphs != 176:
        raise ValueError("expected a native task-switch batch")
    tensors = _task_tensors(labels)
    for current, expected in ((batch[3], tensors[1]), (batch[5], tensors[3]),
                              (batch[7], tensors[5])):
        if not torch.equal(current, expected):
            raise ValueError("batch topology or query roles differ from task-switch protocol")
    return [batch[0].clone(), batch[1].clone(), *tensors]
