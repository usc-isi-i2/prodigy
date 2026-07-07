"""Enrich a retweet-graph artifact with benchmark targets.

Adds, to an existing ``*.pt`` graph artifact:
  * ``node_targets`` / ``node_target_names`` — the profile regression panel
    (skipped for datasets without profile metrics, e.g. cp_hk).
  * ``static_background`` / ``static_holdout`` edge views for static link
    prediction.

Two entry points share one implementation:
  * ``enrich_graph_obj(...)`` — called by the graph generators so future builds
    emit these fields by default.
  * ``main()`` — a standalone CLI that enriches an already-built artifact without
    re-running the (expensive) graph construction.

Env: run with the graph-construction conda env (has ``duckdb`` + ``pyarrow``),
e.g. ``bio-embeddings-v001`` on Tucker.

Example (Tucker):
    python scripts/graph_construction/enrich_graph_targets.py \
        --dataset midterm \
        --graph-path /dataMeR2/phil/data/midterm/graphs/retweet_graph_parquet.pt \
        --parquet-glob '/dataMeR2/phil/data/midterm/parquet/*/*.parquet'
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import torch

try:
    from . import profile_adapters as pa_adapters
    from .benchmark_targets import (
        attach_benchmark_targets,
        build_profile_node_targets,
        build_static_edge_split,
    )
except ImportError:  # running as a plain script
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import profile_adapters as pa_adapters
    from benchmark_targets import (
        attach_benchmark_targets,
        build_profile_node_targets,
        build_static_edge_split,
    )


def parquet_scan_sql(parquet_files) -> str:
    """Build a ``read_parquet([...])`` FROM-expression from a list of paths.

    Convenience for generators that already resolved their input file list and
    want to pass it straight to an adapter as ``scan_sql``.
    """
    files = ", ".join("'" + str(p).replace("'", "''") + "'" for p in parquet_files)
    return f"read_parquet([{files}])"


def _read_edge_index(graph_obj: dict[str, Any]) -> torch.Tensor:
    ei = graph_obj.get("edge_index")
    if ei is None and "data" in graph_obj:
        ei = getattr(graph_obj["data"], "edge_index", None)
    if ei is None:
        raise ValueError("graph artifact has no edge_index")
    return ei.long()


def _fetch_raw_by_user(
    dataset: str,
    user_ids,
    *,
    conn: Any = None,
    scan_sql: Optional[str] = None,
    node_json: Optional[str] = None,
) -> Optional[dict[int, dict[str, Any]]]:
    """Dispatch to the right profile adapter; None when the dataset has no metrics."""
    kind = pa_adapters.ADAPTERS.get(dataset)
    if kind is None:
        return None
    if kind == "flat":
        return pa_adapters.fetch_profiles_flat(conn, scan_sql, user_ids)
    if kind == "covid19":
        return pa_adapters.fetch_profiles_covid19(conn, scan_sql, user_ids)
    if kind == "twibot20":
        return pa_adapters.fetch_profiles_twibot20(node_json, user_ids)
    raise ValueError(f"Unknown adapter kind {kind!r} for dataset {dataset!r}")


def enrich_graph_obj(
    graph_obj: dict[str, Any],
    dataset: str,
    *,
    conn: Any = None,
    scan_sql: Optional[str] = None,
    node_json: Optional[str] = None,
    holdout_frac: float = 0.15,
    seed: int = 0,
    reference_date: Optional[datetime] = None,
) -> dict[str, Any]:
    """Compute + attach benchmark targets into ``graph_obj`` in place.

    Returns a stats dict (also stored under ``graph_obj['benchmark_target_stats']``).
    Regression targets are skipped automatically for datasets without a profile
    adapter (e.g. cp_hk); static link-prediction views are always added.
    """
    user_ids = list(graph_obj.get("user_ids") or [])
    if not user_ids:
        raise ValueError("graph artifact has no user_ids; cannot align node targets")
    edge_index = _read_edge_index(graph_obj)

    stats: dict[str, Any] = {"dataset": dataset}

    node_targets = None
    raw_by_user = _fetch_raw_by_user(
        dataset, user_ids, conn=conn, scan_sql=scan_sql, node_json=node_json
    )
    if raw_by_user is not None:
        node_targets, profile_stats = build_profile_node_targets(
            user_ids, raw_by_user, reference_date=reference_date
        )
        stats["profile"] = profile_stats
    else:
        stats["profile"] = {"skipped": True, "reason": "no profile adapter for dataset"}

    static_split = build_static_edge_split(edge_index, holdout_frac=holdout_frac, seed=seed)
    stats["static_split"] = static_split.stats

    attach_benchmark_targets(
        graph_obj,
        node_targets=node_targets,
        static_split=static_split,
        edge_attr=graph_obj.get("edge_attr"),
        edge_attr_feature_names=graph_obj.get("edge_attr_feature_names"),
    )
    graph_obj["benchmark_target_stats"] = stats
    return stats


def _make_duckdb_conn(threads: int) -> Any:
    import duckdb

    conn = duckdb.connect()
    conn.execute(f"PRAGMA threads={int(threads)}")
    return conn


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=sorted(pa_adapters.ADAPTERS) + ["cp_hk_twitter"],
                   help="Dataset key (selects the profile adapter; cp_hk_twitter = static-LP only).")
    p.add_argument("--graph-path", required=True, help="Input .pt artifact to enrich.")
    p.add_argument("--out", default="", help="Output .pt path (default: overwrite --graph-path).")
    p.add_argument("--parquet-glob", default="", help="Parquet glob for flat/covid19 adapters.")
    p.add_argument("--node-json", default="", help="node.json path for the twibot20 adapter.")
    p.add_argument("--holdout-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reference-date", default="", help="ISO date for account_age_days (optional).")
    p.add_argument("--threads", type=int, default=8)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    graph_path = Path(args.graph_path)
    out_path = Path(args.out) if args.out else graph_path
    reference_date = datetime.fromisoformat(args.reference_date) if args.reference_date else None

    print(f"[enrich] loading {graph_path}")
    graph_obj = torch.load(graph_path, map_location="cpu", weights_only=False)

    conn = None
    scan_sql = None
    if args.dataset in pa_adapters.ADAPTERS and pa_adapters.ADAPTERS[args.dataset] != "twibot20":
        if not args.parquet_glob:
            raise SystemExit(f"--parquet-glob is required for dataset {args.dataset}")
        conn = _make_duckdb_conn(args.threads)
        scan_sql = f"read_parquet('{args.parquet_glob}')"

    stats = enrich_graph_obj(
        graph_obj,
        args.dataset,
        conn=conn,
        scan_sql=scan_sql,
        node_json=(args.node_json or None),
        holdout_frac=args.holdout_frac,
        seed=args.seed,
        reference_date=reference_date,
    )
    print("[enrich] stats:")
    print(json.dumps(stats, indent=2, default=str))

    print(f"[enrich] writing {out_path}")
    torch.save(graph_obj, out_path)

    meta_path = out_path.with_suffix(".benchmark_targets.json")
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, default=str)
    print(f"[enrich] wrote stats sidecar {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
