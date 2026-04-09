#!/usr/bin/env python3
import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]

DATASET_MODULES = {
    "midterm": REPO_ROOT / "data" / "data" / "midterm" / "scripts" / "build_retweet_graph.py",
    "ukr_rus_twitter": REPO_ROOT / "data" / "data" / "ukr_rus_twitter" / "scripts" / "generate_retweet_graph.py",
    "covid19_twitter": REPO_ROOT / "data" / "data" / "covid19_twitter" / "scripts" / "generate_retweet_graph.py",
}


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Summarize a graph artifact and its matching raw data. "
            "The script infers raw_glob and max_files from graph.meta.json when possible."
        )
    )
    p.add_argument(
        "--dataset",
        required=True,
        choices=sorted(DATASET_MODULES.keys()),
    )
    p.add_argument("--graph", required=True, help="Path to *.pt graph artifact")
    p.add_argument(
        "--raw_glob",
        default="",
        help="Optional override for the raw data glob. If omitted, infer from graph.meta.json.",
    )
    p.add_argument(
        "--max_files",
        type=int,
        default=-1,
        help="Optional override for raw max_files. If negative, infer from graph.meta.json.",
    )
    p.add_argument(
        "--format",
        choices=["tsv", "json"],
        default="tsv",
        help="Output format.",
    )
    return p.parse_args()


def load_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_meta(graph_path: Path) -> Dict[str, Any]:
    meta_path = graph_path.with_suffix(".meta.json")
    if not meta_path.exists():
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_raw_source(dataset: str, graph_path: Path, raw_glob: str, max_files: int) -> Tuple[str, int]:
    meta = load_meta(graph_path)
    if raw_glob:
        resolved_glob = raw_glob
    else:
        if dataset == "covid19_twitter":
            resolved_glob = meta.get("json_glob", "")
        else:
            resolved_glob = meta.get("csv_glob", "")

    if not resolved_glob:
        raise ValueError(
            "Could not infer raw_glob from graph.meta.json. "
            "Pass --raw_glob explicitly."
        )

    if max_files >= 0:
        resolved_max_files = max_files
    else:
        resolved_max_files = int(meta.get("max_files", 0) or 0)

    return resolved_glob, resolved_max_files


def count_raw_midterm(module, raw_glob: str, max_files: int) -> Tuple[int, int]:
    import pandas as pd

    raw = module.load_raw_rows(raw_glob, max_files)
    user_ids = pd.to_numeric(raw.get("userid"), errors="coerce")
    n_users = int(user_ids.dropna().nunique())
    return int(len(raw)), n_users


def count_raw_ukr(module, raw_glob: str, max_files: int) -> Tuple[int, int]:
    raw = module.load_raw_rows(raw_glob, max_files)
    handles = module.normalize_handle(raw["screen_name"])
    n_users = int(handles.dropna().nunique())
    return int(len(raw)), n_users


def count_raw_covid(module, raw_glob: str, max_files: int) -> Tuple[int, int]:
    raw = module.load_raw_rows(raw_glob, max_files)
    handles = raw["screen_name"].astype("string").str.strip().str.lower()
    handles = handles.mask(handles.isin(["", "nan", "none", "<na>"]))
    n_users = int(handles.dropna().nunique())
    return int(len(raw)), n_users


def load_graph_tensors(raw: Dict[str, Any]):
    import torch

    x = raw.get("x")
    edge_index = raw.get("edge_index")
    edge_attr = raw.get("edge_attr")

    if x is None or edge_index is None:
        data = raw.get("data")
        if data is None:
            raise KeyError("Graph artifact must contain top-level x/edge_index or a nested data object.")
        x = data.x
        edge_index = data.edge_index
        if edge_attr is None:
            edge_attr = getattr(data, "edge_attr", None)

    return x, edge_index, edge_attr


def summarize_graph(raw: Dict[str, Any]) -> Dict[str, Any]:
    x, edge_index, edge_attr = load_graph_tensors(raw)
    n_nodes = int(x.shape[0])
    n_edges = int(edge_index.shape[1])
    n_node_features = int(x.shape[1])
    n_edge_features = int(edge_attr.shape[1]) if edge_attr is not None else 0
    mean_deg = float(n_edges / n_nodes) if n_nodes else 0.0

    # Interpret centrality as mean degree centrality on the directed graph:
    # average(total_degree / (n_nodes - 1)) = 2E / (N(N-1)).
    if n_nodes > 1:
        mean_centrality = float((2.0 * n_edges) / (n_nodes * (n_nodes - 1)))
    else:
        mean_centrality = 0.0

    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "mean_deg": mean_deg,
        "n_node_features": n_node_features,
        "n_edge_features": n_edge_features,
        "mean_centrality": mean_centrality,
    }


def main():
    args = parse_args()
    graph_path = Path(args.graph).expanduser().resolve()
    raw_glob, max_files = resolve_raw_source(
        args.dataset,
        graph_path,
        args.raw_glob,
        args.max_files,
    )

    module = load_module_from_path(f"summary_{args.dataset}", DATASET_MODULES[args.dataset])
    if args.dataset == "midterm":
        n_tweets, n_users = count_raw_midterm(module, raw_glob, max_files)
    elif args.dataset == "ukr_rus_twitter":
        n_tweets, n_users = count_raw_ukr(module, raw_glob, max_files)
    elif args.dataset == "covid19_twitter":
        n_tweets, n_users = count_raw_covid(module, raw_glob, max_files)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    import torch

    raw_graph = torch.load(graph_path, map_location="cpu")
    graph_summary = summarize_graph(raw_graph)

    row = {
        "dataset": args.dataset,
        "graph": str(graph_path),
        "raw_glob": raw_glob,
        "max_files": int(max_files),
        "n_tweets": int(n_tweets),
        "n_users": int(n_users),
        **graph_summary,
    }

    if args.format == "json":
        print(json.dumps(row, indent=2, sort_keys=True))
    else:
        keys = [
            "dataset",
            "n_tweets",
            "n_users",
            "n_nodes",
            "n_edges",
            "mean_deg",
            "n_node_features",
            "n_edge_features",
            "mean_centrality",
        ]
        print("\t".join(keys))
        print("\t".join(str(row[k]) for k in keys))
        print(
            "# mean_deg is directed mean out-degree (= E / N); "
            "mean_centrality is mean degree centrality (= 2E / (N(N-1)))."
        )


if __name__ == "__main__":
    main()
