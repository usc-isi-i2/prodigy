"""One native FB input-layout fixture, without a model or target predictions.

This is an execution preflight (workers=0, one episode), not the 500-episode
scientific evaluation. It cannot be used as a replacement primary capture.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    # Import helper beside this script without importing project model packages.
    native = importlib.import_module("run_native")
    upstream = native.absolute_path(args.upstream, "upstream")
    root = native.absolute_path(args.root, "root")
    output = native.absolute_path(args.output, "output")
    if output.exists() or output.is_relative_to(root) or output.is_relative_to(upstream):
        raise ValueError("Use a fresh output outside the assets and original source")
    source = native.verify_upstream(upstream)
    plan = {"kind": "layout_preflight_not_performance", "upstream": source,
            "root": str(root), "output": str(output), "ways": 20,
            "shots": 3, "queries": 4, "episodes": 1, "workers": 0,
            "model_forwards": 0, "seed": 0, "execute": args.execute}
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        return
    receipts = native.verify_receipts(root, ["FB15K-237"])
    versions = native.runtime_versions()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    native.write_json(output / "protocol.json", {**plan, "receipts": receipts, "runtime": versions})
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(upstream))
    os.chdir(upstream)
    import random
    import numpy as np
    import torch
    from data.kg import get_kg_dataset, get_kg_dataloader, idx_split
    from data.dataloader import MulticlassTask
    native.install_sampler_compat(MulticlassTask)
    torch.set_num_threads(2)
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    dataset = get_kg_dataset(str(root), "FB15K-237", bert="sentence-transformers/all-mpnet-base-v2",
                             bert_device="cpu", node_graph=False)
    loader = get_kg_dataloader(dataset, task_name="multiway_classification", split="test",
                              node_split="", batch_size=1, n_way=20, n_shot=3, n_query=4,
                              batch_count=1, root=str(root / "FB15K-237"), num_workers=0,
                              aug="", aug_test=False, train_cap=None, linear_probe=False,
                              label_set=set(range(200)), all_test=True, split_labels=False)
    inputs = next(iter(loader))
    graph, x_label, y_true, meta_edges, meta_attrs, query_flags, *_ = inputs
    rows, ways = y_true.shape
    query = query_flags.reshape(rows, ways).bool()
    assert torch.equal(query, query[:, :1].expand_as(query))
    is_query = query[:, 0]
    labels = y_true.argmax(1)
    assert rows == graph.num_graphs == 140 and ways == 20
    assert graph.x.shape[1] == 770 and int(is_query.sum()) == 80
    assert torch.all(torch.bincount(labels[~is_query], minlength=20) == 3)
    assert torch.all(torch.bincount(labels[is_query], minlength=20) == 4)
    assert torch.all(graph.x[graph.ptr[:-1], -1] == 1)
    assert torch.all(graph.x[graph.ptr[:-1] + 1, -2] == 1)
    assert torch.equal(graph.batch[graph.edge_index[0]], graph.batch[graph.edge_index[1]])
    assert graph.edge_attr.shape[0] == graph.edge_index.shape[1]
    # Inspect the actual native task object, not an intended split description.
    task = loader.batch_sampler.task
    support_ids = np.flatnonzero(task.train_label >= 0)
    query_ids = np.flatnonzero(task.labels >= 0)
    original_split = idx_split(len(dataset))
    assert np.intersect1d(support_ids, query_ids).size == 0
    assert np.array_equal(query_ids, np.sort(original_split["valid"]))
    assert np.array_equal(support_ids, np.sort(original_split["train"] + original_split["test"]))
    fixture = output / "inputs.pt"
    torch.save(tuple(inputs), fixture)
    summary = {"kind": "layout_preflight_not_performance", "model_forwards": 0,
               "argument_count": len(inputs), "subgraphs": rows, "ways": ways,
               "support_subgraphs": int((~is_query).sum()), "query_subgraphs": int(is_query.sum()),
               "x_shape": list(graph.x.shape), "edge_shape": list(graph.edge_index.shape),
               "edge_attr_shape": list(graph.edge_attr.shape), "meta_edge_shape": list(meta_edges.shape),
               "meta_attr_shape": list(meta_attrs.shape), "x_label_shape": list(x_label.shape),
               "support_pool_edges": len(support_ids), "query_pool_edges": len(query_ids),
               "pool_overlap": 0, "actual_native_split_verified": True,
               "fixture_sha256": native.file_sha256(fixture),
               "upstream_imports": native.verify_imports(upstream)}
    native.write_json(output / "summary.json", summary)
    native.write_json(output / "execution_status.json", {"status": "complete"})
    print(json.dumps({key: value for key, value in summary.items() if key != "upstream_imports"}, indent=2))


if __name__ == "__main__":
    main()
