#!/usr/bin/env python3
"""Create a compact page-profile input containing only graph participants."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from scripts.graph_construction.facebook_page_reference_nodes import select_page_nodes


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--target-node-count",
        type=int,
        default=0,
        help="Keep all edge participants and add active page-profile isolates to this size.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tables_root = args.tables_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    edges_path = tables_root / "page_reference_edges.parquet"
    profiles_path = tables_root / "page_profiles.parquet"
    for path in (edges_path, profiles_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_root}")
    building_root = output_root.with_name(output_root.name + f".building.{os.getpid()}")
    if building_root.exists():
        raise FileExistsError(f"Refusing to reuse temporary output: {building_root}")
    building_root.mkdir(parents=True)

    edges = pq.read_table(
        edges_path, columns=["source_account_id", "target_account_id"]
    )
    profiles = pq.read_table(profiles_path)
    structural_nodes = set(edges.column(0).to_pylist()) | set(edges.column(1).to_pylist())
    graph_nodes, structural_nodes = select_page_nodes(
        structural_nodes, profiles, args.target_node_count
    )
    mask = pc.is_in(profiles.column("account_id"), value_set=pa.array(graph_nodes, type=pa.string()))
    selected = profiles.filter(mask)
    selected_ids = set(selected.column("account_id").to_pylist())
    missing = sorted(set(graph_nodes) - selected_ids)
    if missing:
        raise ValueError(f"Missing profiles for {len(missing)} graph nodes; examples={missing[:5]}")
    order = pc.sort_indices(selected, sort_keys=[("account_id", "ascending")])
    selected = pc.take(selected, order)
    out_path = building_root / "page_profiles.parquet"
    pq.write_table(selected, out_path, compression="zstd")

    descriptions = selected.column("page_description")
    nonempty_descriptions = int(
        pc.sum(pc.greater(pc.utf8_length(pc.fill_null(descriptions, "")), 0)).as_py()
    )
    summary = {
        "tables_root": str(tables_root),
        "source_edges": str(edges_path),
        "source_profiles": str(profiles_path),
        "graph_nodes": len(graph_nodes),
        "structural_nodes": len(structural_nodes),
        "added_isolated_nodes": len(graph_nodes) - len(structural_nodes),
        "target_node_count": args.target_node_count,
        "selected_profiles": selected.num_rows,
        "profiles_with_description": nonempty_descriptions,
        "description_coverage": nonempty_descriptions / max(1, selected.num_rows),
        "output_sha256": sha256_file(out_path),
    }
    with (building_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    building_root.rename(output_root)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
