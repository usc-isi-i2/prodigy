#!/usr/bin/env python3
"""Build aggregate-provenance global-index exclusions for exact-ID-clean eval.

The generated ``.pt`` files contain graph-row indices, never account IDs.  They
are internal evaluation state and are not publication artifacts.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))

from protocol import DB_TABLE, TARGETS, canonical_global_id, sha256_file  # noqa: E402


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_overlap_ids(conn: Any, target: str) -> set[str]:
    others = [name for name in TARGETS if name != target]
    union = " UNION ".join(f"SELECT id FROM {DB_TABLE[name]}" for name in others)
    query = (
        f"SELECT CAST(id AS VARCHAR) FROM {DB_TABLE[target]} "
        f"INTERSECT SELECT CAST(id AS VARCHAR) FROM ({union}) AS other_ids"
    )
    cursor = conn.execute(query)
    values: set[str] = set()
    while rows := cursor.fetchmany(250_000):
        values.update(str(row[0]) for row in rows)
    return values


def raw_ids(raw: dict[str, Any]) -> list[Any]:
    values = raw.get("raw_user_ids")
    if values is None:
        raise ValueError(
            "merged graph artifact lacks raw_user_ids; refusing namespaced user_ids"
        )
    if len(values) != int(raw["x"].shape[0]):
        raise ValueError("user ID rows do not match graph nodes")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", required=True, type=Path)
    parser.add_argument("--identity-db", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    import duckdb

    print(f"loading graph metadata from {args.graph}", flush=True)
    raw = torch.load(args.graph, map_location="cpu")
    source_names = list(raw.get("source_graph_names", []))
    graph_ids = raw.get("graph_id")
    if graph_ids is None or not isinstance(graph_ids, torch.Tensor):
        raise ValueError("graph artifact lacks tensor graph_id provenance")
    if any(target not in source_names for target in TARGETS):
        raise ValueError(f"exact-ID targets missing from source registry {source_names}")
    ids = raw_ids(raw)
    graph_digest = sha256_file(args.graph)
    identity_db_digest = sha256_file(args.identity_db)
    conn = duckdb.connect(str(args.identity_db), read_only=True)
    summaries = []
    try:
        for target in TARGETS:
            target_id = source_names.index(target)
            target_indices = torch.nonzero(graph_ids == target_id, as_tuple=False).flatten()
            if target_indices.numel() == 0:
                raise ValueError(f"{target}: no graph rows")
            expected_indices = torch.arange(
                int(target_indices[0]), int(target_indices[0]) + int(target_indices.numel())
            )
            if not torch.equal(target_indices, expected_indices):
                raise ValueError(f"{target}: merged source rows are not a contiguous block")
            overlap_ids = load_overlap_ids(conn, target)
            excluded = []
            previous_id = -1
            for index in target_indices.tolist():
                canonical = canonical_global_id(ids[index])
                numeric = int(canonical)
                if numeric <= previous_id:
                    raise ValueError(
                        f"{target}: raw_user_ids are not strictly increasing and unique"
                    )
                previous_id = numeric
                if canonical in overlap_ids:
                    excluded.append(index)
            target_table_count = int(
                conn.execute(f"SELECT count(*) FROM {DB_TABLE[target]}").fetchone()[0]
            )
            if target_table_count != int(target_indices.numel()):
                raise AssertionError(
                    f"{target}: identity DB has {target_table_count} IDs but graph has "
                    f"{target_indices.numel()} nodes"
                )
            if len(excluded) != len(overlap_ids):
                raise AssertionError(
                    f"{target}: mapped {len(excluded)} of {len(overlap_ids)} union-overlap IDs"
                )
            excluded_tensor = torch.tensor(excluded, dtype=torch.long)
            comparison_sources = [name for name in TARGETS if name != target]
            payload = {
                "protocol": "exact_id_exclusion_union3_v1",
                "target": target,
                "comparison_sources": comparison_sources,
                "source_graph_names": source_names,
                "graph_path": str(args.graph),
                "graph_sha256": graph_digest,
                "identity_db_path": str(args.identity_db),
                "identity_db_sha256": identity_db_digest,
                "target_graph_nodes": int(target_indices.numel()),
                "excluded_node_count": int(excluded_tensor.numel()),
                "excluded_node_indices": excluded_tensor,
                "created_utc": datetime.now(timezone.utc).isoformat(),
            }
            output = args.output_dir / f"{target}.pt"
            torch.save(payload, output)
            summary = {
                key: payload[key]
                for key in (
                    "protocol", "target", "comparison_sources", "target_graph_nodes",
                    "excluded_node_count", "graph_sha256", "identity_db_sha256",
                )
            }
            summary["artifact_sha256"] = sha256_file(output)
            summaries.append(summary)
            print(
                f"target={target} nodes={target_indices.numel()} "
                f"excluded={excluded_tensor.numel()} artifact={output}",
                flush=True,
            )
            del overlap_ids, excluded, excluded_tensor, target_indices
    finally:
        conn.close()
    atomic_json(
        args.output_dir / "summary.json",
        {
            "protocol": "exact_id_exclusion_union3_v1",
            "privacy": "aggregate summary only; internal pt files contain graph-row indices, not IDs",
            "targets": summaries,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
