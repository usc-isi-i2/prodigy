import csv
from pathlib import Path

import numpy as np


class TrainingRoleCounter:
    """Accumulate exact node exposure by role from collated NM training batches."""

    def __init__(self, num_nodes):
        self.num_nodes = int(num_nodes)
        self.anchor = np.zeros(self.num_nodes, dtype=np.int64)
        self.support = np.zeros(self.num_nodes, dtype=np.int64)
        self.query = np.zeros(self.num_nodes, dtype=np.int64)
        self.steps = 0

    @staticmethod
    def _as_numpy(value):
        return value.detach().cpu().numpy()

    def observe_batch(self, batch):
        graph = batch[0]
        centers = getattr(graph, "center_node_idx", None)
        anchors = getattr(graph, "task_label_map", None)
        if centers is None or anchors is None:
            raise ValueError(
                "Training role counting requires neighbor-matching batches with "
                "center_node_idx and task_label_map."
            )

        centers = self._as_numpy(centers).reshape(-1).astype(np.int64, copy=False)
        anchors = self._as_numpy(anchors).reshape(-1).astype(np.int64, copy=False)
        edge_query_mask = self._as_numpy(batch[5]).reshape(len(centers), -1)
        query_mask = edge_query_mask[:, 0].astype(bool, copy=False)

        for name, ids in (
            ("anchor", anchors),
            ("support", centers[~query_mask]),
            ("query", centers[query_mask]),
        ):
            if ids.size and (ids.min() < 0 or ids.max() >= self.num_nodes):
                raise ValueError(
                    f"{name} node id outside [0, {self.num_nodes}): "
                    f"min={ids.min()}, max={ids.max()}"
                )
            np.add.at(getattr(self, name), ids, 1)
        self.steps += 1

    def state_dict(self):
        return {
            "num_nodes": self.num_nodes,
            "anchor": self.anchor.copy(),
            "support": self.support.copy(),
            "query": self.query.copy(),
            "steps": self.steps,
        }

    def load_state_dict(self, state):
        if int(state["num_nodes"]) != self.num_nodes:
            raise ValueError(
                f"Role-count node count changed: checkpoint={state['num_nodes']}, "
                f"current={self.num_nodes}."
            )
        for name in ("anchor", "support", "query"):
            values = np.asarray(state[name], dtype=np.int64)
            if values.shape != (self.num_nodes,):
                raise ValueError(f"Invalid saved {name} count shape: {values.shape}.")
            getattr(self, name)[:] = values
        self.steps = int(state["steps"])

    def write_csv(self, path, graph_id=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        graph_ids = None
        if graph_id is not None:
            graph_ids = self._as_numpy(graph_id).reshape(-1)
            if len(graph_ids) != self.num_nodes:
                graph_ids = None

        temporary = path.with_suffix(path.suffix + ".tmp")
        fields = ["node_id"]
        if graph_ids is not None:
            fields.append("graph_id")
        fields += ["anchor_count", "support_count", "query_count", "total_count"]
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(fields)
            for node_id in range(self.num_nodes):
                counts = (
                    int(self.anchor[node_id]),
                    int(self.support[node_id]),
                    int(self.query[node_id]),
                )
                row = [node_id]
                if graph_ids is not None:
                    row.append(int(graph_ids[node_id]))
                writer.writerow([*row, *counts, sum(counts)])
        temporary.replace(path)

