"""Compact IDs for training batches actually consumed, not loader prefetches."""
import gzip
import hashlib
import json
from pathlib import Path

import torch


def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def episode_record(batch, step):
    graph = batch[0]
    if graph.x.device.type != "cpu":
        raise ValueError("record consumed batches before moving them to the GPU")
    anchors = graph.task_label_map.long()
    if anchors.ndim != 2:
        raise ValueError("expected a fixed-way episode label map")
    n_tasks, n_way = anchors.shape
    ids = graph.global_node_ids[graph.ptr[:-1]].long()
    task_ids = graph.task_id_per_sample.long()
    labels = batch[2].argmax(1)
    query = batch[5].reshape(-1, n_way)[:, 0].bool()
    if len(ids) % (n_tasks * n_way):
        raise ValueError("expected equal-size pseudo-classes")
    n_members = len(ids) // (n_tasks * n_way)
    expected_tasks = torch.arange(n_tasks).repeat_interleave(n_way * n_members)
    expected_labels = torch.arange(n_way).repeat_interleave(n_members).repeat(n_tasks)
    if not torch.equal(task_ids, expected_tasks) or not torch.equal(labels, expected_labels):
        raise ValueError("unexpected collator order; do not silently reinterpret members")
    members = ids.reshape(n_tasks, n_way, n_members)
    role_grid = query.reshape(n_tasks, n_way, n_members)
    if not torch.all(role_grid == role_grid[0, 0]):
        raise ValueError("expected one common support/query role pattern")
    nodes = (graph.ptr[1:] - graph.ptr[:-1] - 1).reshape(n_tasks, n_way, n_members)
    # Member sets are retained separately from roles for matched factorial checks.
    return {"step": int(step), "anchor_ids": anchors.tolist(), "member_ids": members.tolist(),
            "query_roles": role_grid[0, 0].int().tolist(), "context_node_counts": nodes.tolist(),
            "source_ids": graph.source_id_per_task.tolist(),
            "anchor_sha256": digest(anchors), "member_order_sha256": digest(members),
            "member_set_sha256": digest(members.sort(-1).values)}


def append_episode_audit(batch, step, logging_dir):
    row = episode_record(batch, step)
    # A complete gzip member per consumed batch keeps prior records readable if
    # a process fails; no unconsumed prefetches are written. Artifacts stay in log/.
    with gzip.open(Path(logging_dir) / "consumed_episodes.jsonl.gz", "at", compresslevel=1) as handle:
        handle.write(json.dumps(row, separators=(",", ":")) + "\n")
