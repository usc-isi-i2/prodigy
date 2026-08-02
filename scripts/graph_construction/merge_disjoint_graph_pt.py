"""Merge multiple saved graph `.pt` artifacts as disjoint components.

This is intentionally a block-concat merge: node indices from each input graph
are offset into one global node space, and no attempt is made to deduplicate
users or aggregate cross-dataset edges.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any

import torch

try:
    import yaml
except ImportError:  # pragma: no cover - depends on environment.
    yaml = None

try:
    from torch_geometric.data import Data as PyGData
except ImportError:  # pragma: no cover - depends on environment.
    PyGData = None


def _log(message: str) -> None:
    print(f"[merge] {message}", flush=True)


def _make_data_object(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor | None,
    y: torch.Tensor,
) -> Any:
    if PyGData is not None:
        return PyGData(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    return SimpleNamespace(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def _parse_input_spec(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        name, path = spec.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Input name is empty in --input {spec!r}")
        return name, Path(path)
    path = Path(spec)
    return path.stem, path


def _load_config(config_path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to load the merge config.")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise TypeError(f"{config_path} must contain a YAML mapping.")
    return config


def _parse_config_inputs(config: dict[str, Any]) -> tuple[list[str], list[Path]]:
    inputs = config.get("inputs")
    if not inputs:
        raise ValueError("Config must define a non-empty 'inputs' field.")

    specs: list[tuple[str, Path]] = []
    if isinstance(inputs, dict):
        for name, path in inputs.items():
            specs.append((str(name), Path(str(path))))
    elif isinstance(inputs, list):
        for entry in inputs:
            if isinstance(entry, str):
                specs.append(_parse_input_spec(entry))
            elif isinstance(entry, dict):
                name = entry.get("name")
                path = entry.get("path")
                if not name or not path:
                    raise ValueError("Each input mapping must contain 'name' and 'path'.")
                specs.append((str(name), Path(str(path))))
            else:
                raise TypeError("Each input entry must be a string or mapping.")
    else:
        raise TypeError("Config 'inputs' must be a mapping or list.")

    names = [name for name, _ in specs]
    if len(set(names)) != len(names):
        raise ValueError(f"Input names must be unique, got {names}")
    return names, [path for _, path in specs]


def _load_graph(name: str, path: Path) -> dict[str, Any]:
    _log(f"loading {name}: {path}")
    raw = torch.load(path, map_location="cpu")
    if not isinstance(raw, dict):
        raise TypeError(f"{path} must contain a dict graph artifact, got {type(raw).__name__}")
    for key in ("x", "edge_index"):
        if key not in raw:
            raise KeyError(f"{path} is missing required key {key!r}")
    if raw["x"].dim() != 2:
        raise ValueError(f"{path}: x must be 2D, got shape {tuple(raw['x'].shape)}")
    if raw["edge_index"].dim() != 2 or raw["edge_index"].shape[0] != 2:
        raise ValueError(f"{path}: edge_index must have shape [2, E]")
    edge_attr = raw.get("edge_attr")
    if edge_attr is not None and edge_attr.shape[0] != raw["edge_index"].shape[1]:
        raise ValueError(f"{path}: edge_attr rows must match edge_index columns")
    return raw


def _ensure_static_split_views(
    graph: dict[str, Any],
    *,
    name: str,
    holdout_frac: float,
    seed: int,
) -> None:
    """Derive static views in memory without rewriting a shared source artifact."""
    background = graph.get("edge_index_views", {}).get("static_background")
    holdout = graph.get("target_edge_index_views", {}).get("static_holdout")
    if (background is None) != (holdout is None):
        raise ValueError(
            f"{name}: only one of static_background/static_holdout exists; "
            "refusing to replace a partial stored split."
        )
    if background is not None:
        _log(f"{name}: using stored static_background/static_holdout views")
        return
    try:
        from scripts.graph_construction.benchmark_targets import build_static_edge_split
    except ModuleNotFoundError:  # direct execution from scripts/graph_construction
        from benchmark_targets import build_static_edge_split

    split = build_static_edge_split(
        graph["edge_index"], holdout_frac=holdout_frac, seed=seed
    )
    graph.setdefault("edge_index_views", {})[
        "static_background"
    ] = split.background_edge_index
    graph.setdefault("target_edge_index_views", {})[
        "static_holdout"
    ] = split.holdout_edge_index
    graph.setdefault("benchmark_target_stats", {})["static_split"] = split.stats
    _log(
        f"{name}: derived canonical static split in memory "
        f"(seed={seed}, holdout_frac={holdout_frac}); source file unchanged"
    )


def _same_list(graphs: list[dict[str, Any]], key: str, default: list[Any] | None = None) -> list[Any]:
    values = [list(graph.get(key, default or [])) for graph in graphs]
    first = values[0]
    for idx, value in enumerate(values[1:], start=1):
        if value != first:
            raise ValueError(f"Input {idx} has different {key}; refusing to merge incompatible artifacts.")
    return first


def _same_mapping(graphs: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [dict(graph.get(key, {})) for graph in graphs]
    first = values[0]
    for idx, value in enumerate(values[1:], start=1):
        if value != first:
            raise ValueError(f"Input {idx} has different {key}; refusing to merge incompatible artifacts.")
    return first


def _offset_edge_index(edge_index: torch.Tensor, offset: int) -> torch.Tensor:
    if edge_index.numel() == 0:
        return edge_index.clone()
    return edge_index.long() + int(offset)


def _concat_optional_tensors(tensors: list[torch.Tensor | None], key: str) -> torch.Tensor | None:
    present = [tensor is not None for tensor in tensors]
    if not any(present):
        return None
    if not all(present):
        raise ValueError(f"Only some inputs contain {key}; refusing an ambiguous partial merge.")
    dims = {tuple(tensor.shape[1:]) for tensor in tensors if tensor is not None}
    if len(dims) != 1:
        raise ValueError(f"Inputs have incompatible {key} trailing dimensions: {sorted(dims)}")
    return torch.cat([tensor for tensor in tensors if tensor is not None], dim=0)


def _merge_edge_index_views(
    graphs: list[dict[str, Any]],
    offsets: list[int],
    views_key: str,
    selected_views: list[str] | None = None,
) -> dict[str, torch.Tensor]:
    if selected_views is not None:
        requested = list(dict.fromkeys(str(view) for view in selected_views))
        for idx, graph in enumerate(graphs):
            missing = [
                view for view in requested
                if view not in graph.get(views_key, {})
            ]
            if missing:
                raise ValueError(
                    f"Input {idx} is missing requested {views_key} views {missing}."
                )
        return {
            view_name: torch.cat(
                [
                    _offset_edge_index(graph[views_key][view_name], offset)
                    for graph, offset in zip(graphs, offsets)
                ],
                dim=1,
            )
            for view_name in requested
        }

    view_sets = [set(graph.get(views_key, {}).keys()) for graph in graphs]
    if not any(view_sets):
        return {}
    first = view_sets[0]
    for idx, view_set in enumerate(view_sets[1:], start=1):
        if view_set != first:
            raise ValueError(
                f"Input {idx} has different {views_key} keys; "
                "merge inputs with matching views or rebuild views after merging."
            )
    merged: dict[str, torch.Tensor] = {}
    for view_name in sorted(first):
        parts = [
            _offset_edge_index(graph[views_key][view_name], offset)
            for graph, offset in zip(graphs, offsets)
        ]
        merged[view_name] = torch.cat(parts, dim=1)
    return merged


def _merge_edge_attr_views(graphs: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    view_sets = [set(graph.get("edge_attr_views", {}).keys()) for graph in graphs]
    if not any(view_sets):
        return {}
    first = view_sets[0]
    for idx, view_set in enumerate(view_sets[1:], start=1):
        if view_set != first:
            raise ValueError(f"Input {idx} has different edge_attr_views keys.")
    merged: dict[str, torch.Tensor] = {}
    for view_name in sorted(first):
        tensors = [graph["edge_attr_views"][view_name] for graph in graphs]
        merged[view_name] = torch.cat(tensors, dim=0)
    return merged


def _namespace_user_ids(names: list[str], graphs: list[dict[str, Any]]) -> tuple[list[str], list[Any]]:
    user_ids: list[str] = []
    raw_user_ids: list[Any] = []
    for name, graph in zip(names, graphs):
        source_user_ids = list(graph.get("user_ids", range(graph["x"].shape[0])))
        if len(source_user_ids) != graph["x"].shape[0]:
            raise ValueError(f"{name}: len(user_ids) does not match x rows.")
        raw_user_ids.extend(source_user_ids)
        user_ids.extend(f"{name}:{user_id}" for user_id in source_user_ids)
    return user_ids, raw_user_ids


def _merge_graphs(
    names: list[str],
    paths: list[Path],
    graphs: list[dict[str, Any]],
    drop_edge_features: bool = False,
    preserve_edge_views: list[str] | None = None,
    preserve_target_edge_views: list[str] | None = None,
) -> dict[str, Any]:
    feature_names = _same_list(graphs, "feature_names")
    if drop_edge_features:
        # Heterogeneous edge schemas (e.g. social graphs' 2-d [rt_weight, mn_weight]
        # vs twitter's 1-d [n_retweets]): drop edge_attr + all edge/label views and
        # keep structure only. Loader tolerates missing edge_attr (edge_view=default
        # falls back to the main edge_index).
        edge_attr_feature_names: list[Any] = []
        label_names: list[Any] = []
    else:
        edge_attr_feature_names = _same_list(graphs, "edge_attr_feature_names")
        label_names = _same_list(graphs, "label_names")

    x_dims = {int(graph["x"].shape[1]) for graph in graphs}
    if len(x_dims) != 1:
        raise ValueError(f"Inputs have incompatible x dimensions: {sorted(x_dims)}")

    node_counts = [int(graph["x"].shape[0]) for graph in graphs]
    edge_counts = [int(graph["edge_index"].shape[1]) for graph in graphs]
    offsets: list[int] = []
    running_nodes = 0
    for node_count in node_counts:
        offsets.append(running_nodes)
        running_nodes += node_count

    x = torch.cat([graph["x"] for graph in graphs], dim=0)
    edge_index = torch.cat(
        [_offset_edge_index(graph["edge_index"], offset) for graph, offset in zip(graphs, offsets)],
        dim=1,
    )

    y_parts = []
    for graph in graphs:
        y = graph.get("y")
        if y is None:
            y = torch.full((graph["x"].shape[0],), -1, dtype=torch.long)
        y_parts.append(y)
    y = torch.cat(y_parts, dim=0)

    if drop_edge_features:
        edge_attr = None
        edge_index_views = _merge_edge_index_views(
            graphs, offsets, "edge_index_views", preserve_edge_views
        ) if preserve_edge_views else {}
        target_edge_index_views = _merge_edge_index_views(
            graphs, offsets, "target_edge_index_views", preserve_target_edge_views
        ) if preserve_target_edge_views else {}
        edge_attr_views: dict[str, torch.Tensor] = {}
        edge_attr_feature_names_views: dict[str, Any] = {}
    else:
        edge_attr = _concat_optional_tensors([graph.get("edge_attr") for graph in graphs], "edge_attr")
        edge_index_views = _merge_edge_index_views(graphs, offsets, "edge_index_views")
        target_edge_index_views = _merge_edge_index_views(graphs, offsets, "target_edge_index_views")
        edge_attr_views = _merge_edge_attr_views(graphs)
        edge_attr_feature_names_views = _same_mapping(graphs, "edge_attr_feature_names_views")

    user_ids, raw_user_ids = _namespace_user_ids(names, graphs)
    handles: list[Any] = []
    for graph in graphs:
        source_handles = list(graph.get("handles", [None] * graph["x"].shape[0]))
        if len(source_handles) != graph["x"].shape[0]:
            source_handles = [None] * graph["x"].shape[0]
        handles.extend(source_handles)

    graph_id = torch.cat(
        [torch.full((node_count,), idx, dtype=torch.long) for idx, node_count in enumerate(node_counts)],
        dim=0,
    )
    source_node_offsets = {name: offset for name, offset in zip(names, offsets)}
    source_edge_offsets: dict[str, int] = {}
    running_edges = 0
    for name, edge_count in zip(names, edge_counts):
        source_edge_offsets[name] = running_edges
        running_edges += edge_count

    data = _make_data_object(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.feature_names = feature_names
    data.edge_attr_feature_names = edge_attr_feature_names
    data.user_ids = user_ids
    data.graph_id = graph_id

    merged: dict[str, Any] = {
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "edge_attr_feature_names": edge_attr_feature_names,
        "user_ids": user_ids,
        "raw_user_ids": raw_user_ids,
        "u2i": {user_id: idx for idx, user_id in enumerate(user_ids)},
        "feature_names": feature_names,
        "handles": handles,
        "y": y,
        "label_names": label_names,
        "data": data,
        "graph_id": graph_id,
        "source_graph_names": names,
        "source_graph_paths": [path.as_posix() for path in paths],
        "source_node_offsets": source_node_offsets,
        "source_edge_offsets": source_edge_offsets,
        "source_node_counts": dict(zip(names, node_counts)),
        "source_edge_counts": dict(zip(names, edge_counts)),
        "merge_policy": "disjoint_block_concat",
        "preserved_edge_views": list(preserve_edge_views or []),
        "preserved_target_edge_views": list(preserve_target_edge_views or []),
        "source_static_split_stats": {
            name: graph.get(
                "static_split_stats",
                graph.get("benchmark_target_stats", {}).get("static_split"),
            )
            for name, graph in zip(names, graphs)
        },
    }
    if edge_index_views:
        merged["edge_index_views"] = edge_index_views
    if edge_attr_views:
        merged["edge_attr_views"] = edge_attr_views
    if edge_attr_feature_names_views:
        merged["edge_attr_feature_names_views"] = edge_attr_feature_names_views
    if target_edge_index_views:
        merged["target_edge_index_views"] = target_edge_index_views
        if "temporal_new" in target_edge_index_views:
            merged["future_edge_index"] = target_edge_index_views["temporal_new"]

    return merged


def _validate_merged(graph: dict[str, Any]) -> None:
    x = graph["x"]
    edge_index = graph["edge_index"]
    edge_attr = graph.get("edge_attr")
    if len(graph["user_ids"]) != x.shape[0]:
        raise ValueError("len(user_ids) must equal x rows.")
    if edge_index.numel() and int(edge_index.max()) >= x.shape[0]:
        raise ValueError("edge_index contains an out-of-range node index.")
    if edge_attr is not None and edge_attr.shape[0] != edge_index.shape[1]:
        raise ValueError("edge_attr rows must equal edge count.")
    if torch.isnan(x).any():
        raise ValueError("x contains NaN.")
    for views_key in ("edge_index_views", "target_edge_index_views"):
        for view_name, view in graph.get(views_key, {}).items():
            if view.dim() != 2 or view.shape[0] != 2:
                raise ValueError(
                    f"{views_key}[{view_name!r}] must have shape [2, E]."
                )
            if view.numel() and int(view.max()) >= x.shape[0]:
                raise ValueError(
                    f"{views_key}[{view_name!r}] contains an out-of-range node index."
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge graph .pt files as disjoint components.")
    parser.add_argument(
        "config",
        help="YAML config path with 'inputs' and 'out'.",
    )
    return parser.parse_args()


def main() -> None:
    started = time.monotonic()
    args = parse_args()
    config_path = Path(args.config)
    config = _load_config(config_path)
    names, paths = _parse_config_inputs(config)
    out = config.get("out")
    if not out:
        raise ValueError("Config must define 'out'.")

    graphs = [_load_graph(name, path) for name, path in zip(names, paths)]

    if bool(config.get("build_static_split_if_missing", False)):
        split_seed = int(config.get("static_split_seed", 0))
        holdout_frac = float(config.get("static_holdout_frac", 0.15))
        for name, graph in zip(names, graphs):
            _ensure_static_split_views(
                graph,
                name=name,
                holdout_frac=holdout_frac,
                seed=split_seed,
            )

    drop_edge_features = bool(config.get("drop_edge_features", False))
    preserve_edge_views = [str(v) for v in config.get("preserve_edge_views", [])]
    preserve_target_edge_views = [
        str(v) for v in config.get("preserve_target_edge_views", [])
    ]
    if drop_edge_features:
        kept = preserve_edge_views + preserve_target_edge_views
        suffix = f"; preserving requested structural views {kept}" if kept else ""
        _log(
            "drop_edge_features=True: dropping edge_attr + unrequested edge/label views"
            + suffix
        )
    _log("merging inputs as disjoint components")
    merged = _merge_graphs(
        names,
        paths,
        graphs,
        drop_edge_features=drop_edge_features,
        preserve_edge_views=preserve_edge_views,
        preserve_target_edge_views=preserve_target_edge_views,
    )
    _validate_merged(merged)

    out_path = Path(str(out))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _log(f"writing merged artifact: {out_path}")
    torch.save(merged, out_path)

    meta = {
        "merge_policy": merged["merge_policy"],
        "config_path": config_path.as_posix(),
        "source_graphs": [
            {
                "name": name,
                "path": path.as_posix(),
                "node_offset": int(merged["source_node_offsets"][name]),
                "edge_offset": int(merged["source_edge_offsets"][name]),
                "nodes": int(merged["source_node_counts"][name]),
                "edges": int(merged["source_edge_counts"][name]),
            }
            for name, path in zip(names, paths)
        ],
        "nodes": int(merged["x"].shape[0]),
        "edges": int(merged["edge_index"].shape[1]),
        "node_feature_dim": int(merged["x"].shape[1]),
        "edge_feature_names": merged.get("edge_attr_feature_names", []),
        "edge_views": {
            name: int(view.shape[1])
            for name, view in merged.get("edge_index_views", {}).items()
        },
        "target_edge_views": {
            name: int(view.shape[1])
            for name, view in merged.get("target_edge_index_views", {}).items()
        },
        "source_static_split_stats": merged.get("source_static_split_stats", {}),
        "static_split_seed": int(config.get("static_split_seed", 0)),
        "static_holdout_frac": float(config.get("static_holdout_frac", 0.15)),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    meta_path = out_path.with_suffix(".meta.json")
    _log(f"writing metadata: {meta_path}")
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

    _log(
        "done: "
        f"{meta['nodes']:,} nodes, {meta['edges']:,} edges, "
        f"{meta['elapsed_seconds']:.1f}s"
    )


if __name__ == "__main__":
    main()
