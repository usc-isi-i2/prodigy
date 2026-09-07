"""Tensor-only portable input schema, with explicit paths and immutable inputs."""
import hashlib
import json
from pathlib import Path

import torch
from torch_geometric.data import Batch

GRAPH_FIELDS = ('x', 'edge_index', 'edge_attr', 'supernode', 'edge_index_supernode',
                'ptr', 'batch', 'global_node_ids', 'task_id_per_sample', 'task_label_map')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_path(root, record):
    candidate = (root / record['path']).resolve()
    if not candidate.is_relative_to(root.resolve()) or not candidate.is_file():
        raise ValueError('Input path escapes the supplied input directory or is missing')
    if sha(candidate) != record['sha256']:
        raise ValueError('Input file hash mismatch: '+record['path'])
    return candidate


def pack_batch(batch):
    g = batch[0]
    values = {k: g[k].detach().cpu().clone() if isinstance(g[k], torch.Tensor) else g[k] for k in GRAPH_FIELDS}
    # Only virtual/real membership is needed by the interventions. Discard account IDs.
    real = values['global_node_ids'] >= 0
    values['global_node_ids'] = torch.where(real, torch.arange(len(real)), -1)
    if len(batch) not in (9, 10) or any(not isinstance(v, torch.Tensor) for v in batch[1:]):
        raise ValueError('Expected production tensor batch arguments')
    return {'format': 1, 'graph': values, 'arguments': [v.detach().cpu().clone() for v in batch[1:]]}


def unpack_batch(record):
    if record['format'] != 1 or set(record['graph']) != set(GRAPH_FIELDS):
        raise ValueError('Unsupported tensor input schema')
    if any(v is not None and not isinstance(v, torch.Tensor) for v in record['graph'].values()):
        raise ValueError('Graph fields must be tensors or None')
    g = Batch(**record['graph'])
    batch = [g] + record['arguments']
    if len(batch) not in (9, 10) or any(not isinstance(v, torch.Tensor) for v in batch[1:]):
        raise ValueError('Non-tensor model arguments')
    if batch[2].ndim != 2 or batch[2].shape[1] != 2:
        raise ValueError('Binary episodes required')
    n = len(g.ptr)-1
    if batch[2].shape[0] != n or len(g.supernode) != n or len(g.task_id_per_sample) != n:
        raise ValueError('Graph/example cardinality mismatch')
    if g.ptr[0] != 0 or g.ptr[-1] != len(g.x) or not torch.all(g.ptr.diff() > 0):
        raise ValueError('Invalid subgraph boundaries')
    if not torch.equal(g.batch, torch.repeat_interleave(torch.arange(n), g.ptr.diff())):
        raise ValueError('Invalid subgraph membership')
    if not torch.isfinite(g.x).all():
        raise ValueError('Nonfinite input features')
    return batch


def load_batch(root, record):
    return unpack_batch(torch.load(checked_path(root, record), map_location='cpu', weights_only=True))


def labels_for(batch, use_global):
    q = batch[5].reshape(-1, 2)[:, 0].bool()
    return {'local_y': batch[2][q].argmax(1).long(),
            'mapping': batch[0].task_label_map[batch[0].task_id_per_sample[q]].long(),
            'use_global': use_global}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
