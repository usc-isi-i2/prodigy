"""Read-only input-degree/label descriptors; labels never choose interventions."""
import argparse
import csv
import json
from pathlib import Path

import torch

from .replay import batch_hash
from .role_context import query_mask


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--reference-replay", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() or torch.cuda.is_available():
        raise ValueError("new output and hidden GPUs required")
    torch.set_num_threads(1)
    protocol = json.loads((args.reference_replay / "protocol.json").read_text())
    rows = []
    for target in protocol["targets"]:
        original = Path(protocol["cache_roots"][f"{target}/original"])
        graph_path = json.loads((original / target / "cache.json").read_text())["graph_path"]
        raw = torch.load(graph_path, map_location="cpu", weights_only=False)
        labels = (raw["y"] if isinstance(raw, dict) else raw.y).reshape(-1).clone()
        del raw
        for stream in ("original", "fresh"):
            root = Path(protocol["cache_roots"][f"{target}/{stream}"]) / target
            cache = json.loads((root / "cache.json").read_text())
            totals = {role: {k: 0 for k in ("real_node_occurrences", "in_degree_zero", "in_degree_one", "in_degree_over_one",
                                            "subgraphs", "center_in_zero", "center_in_one", "center_in_over_one",
                                            "edges", "labeled_edges", "same_label_edges")}
                      for role in ("query", "support")}
            for bi in range(32):
                batch = torch.load(root / "batches" / f"batch_{bi:03d}.pt", map_location="cpu", weights_only=False)
                if batch_hash(batch) != cache["batch_sha256"][bi]:
                    raise ValueError("input fingerprint mismatch")
                g, q = batch[0], query_mask(batch)
                node_ids, centers = g.global_node_ids, g.ptr[:-1]
                real = node_ids >= 0
                node_labels = torch.full((len(node_ids),), -1, dtype=labels.dtype)
                node_labels[real] = labels[node_ids[real]]
                observed = g.task_label_map[g.task_id_per_sample, batch[2].argmax(1)]
                if not torch.equal(node_labels[centers], observed):
                    raise ValueError("raw labels and evaluated centers do not align")
                inc = torch.bincount(g.edge_index[1], minlength=len(g.x))
                for role, selected in (("query", q), ("support", ~q)):
                    t = totals[role]
                    values = inc[real & selected[g.batch]]
                    center_values = inc[centers[selected]]
                    edge = g.edge_index[:, selected[g.batch[g.edge_index[0]]]]
                    edge_y = node_labels[edge]
                    valid = (edge_y >= 0).all(0)
                    t["real_node_occurrences"] += len(values)
                    t["in_degree_zero"] += int((values == 0).sum())
                    t["in_degree_one"] += int((values == 1).sum())
                    t["in_degree_over_one"] += int((values > 1).sum())
                    t["subgraphs"] += len(center_values)
                    t["center_in_zero"] += int((center_values == 0).sum())
                    t["center_in_one"] += int((center_values == 1).sum())
                    t["center_in_over_one"] += int((center_values > 1).sum())
                    t["edges"] += edge.shape[1]
                    t["labeled_edges"] += int(valid.sum())
                    t["same_label_edges"] += int(((edge_y[0] == edge_y[1]) & valid).sum())
            for role, values in totals.items():
                rows.append({"target": target, "stream": stream, "role": role, **values,
                             "fraction_in_degree_over_one": values["in_degree_over_one"]/values["real_node_occurrences"],
                             "edge_label_coverage": values["labeled_edges"]/values["edges"] if values["edges"] else None,
                             "labeled_edge_agreement": values["same_label_edges"]/values["labeled_edges"] if values["labeled_edges"] else None})
    args.output.mkdir(parents=True)
    with (args.output / "input_degrees_labels.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "scope.json").write_text(json.dumps({"rows": len(rows), "description": "Occurrence-weighted descriptors of cached sampled subgraphs, not full-network or training-source homophily. Raw labels are used only for post-hoc descriptors and checked against evaluated centers. No model intervention uses these labels."}, indent=2)+"\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
