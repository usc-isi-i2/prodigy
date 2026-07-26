"""Build a midterm Twitter retweet graph directly from staged parquet files.

This mirrors the other parquet graph builders: graph events and edge aggregation
stay in DuckDB, and node features come from the bio embedding store.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pyarrow as pa
import torch

try:
    from torch_geometric.data import Data as PyGData
except ImportError:  # pragma: no cover - depends on environment.
    PyGData = None


DEFAULT_PARQUET_ROOT = "/dataMeR1/phil/data/midterm/parquet"
DEFAULT_BIO_EMBEDDINGS_ROOT = (
    "/dataMeR1/phil/data/midterm/bio_embeddings/"
    "gte-multilingual-base/version=v001"
)
DEFAULT_OUT = "data/data/midterm/graphs/retweet_graph_parquet.pt"
DEFAULT_HISTORY_FRACTION = 0.3
EDGE_ATTR_FEATURE_NAMES = ["n_retweets"]
TIMESTAMP_CANDIDATES = [
    "observed_at",
    "timestamp",
    "created_ts",
    "created_time",
    "created",
    "tweet_created_at",
    "created_at",
    "date",
]


def _log_progress(message: str) -> None:
    print(f"[progress] {message}", flush=True)


def _format_elapsed(seconds: float) -> str:
    return f"{seconds:.1f}s"


def _require_duckdb() -> Any:
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - depends on environment.
        raise RuntimeError(
            "duckdb is required to build the parquet retweet graph. "
            "Install the dataset environment with duckdb, pyarrow, and torch."
        ) from exc
    return duckdb


def _make_data_object(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    y: torch.Tensor,
) -> Any:
    if PyGData is not None:
        return PyGData(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    return SimpleNamespace(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def _sql_literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _sql_path_list(paths: list[str]) -> str:
    return "[" + ", ".join(_sql_literal(path) for path in paths) + "]"


def _quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a directed midterm retweet graph from staged parquet input "
            "with bio-embedding node features."
        )
    )
    parser.add_argument("--config", default="", help="Optional YAML config to load before CLI overrides.")
    parser.add_argument("--parquet-root", default=DEFAULT_PARQUET_ROOT)
    parser.add_argument(
        "--parquet-path",
        action="append",
        default=[],
        help="Optional parquet file or directory path. Repeatable. Defaults to --parquet-root recursively.",
    )
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--bio-embeddings-root", default=DEFAULT_BIO_EMBEDDINGS_ROOT)
    parser.add_argument(
        "--graph-cutoff",
        default="",
        help=(
            "Optional inclusive cutoff timestamp. Retweet edges are built from rows observed at or before "
            "this cutoff, and bio features use the latest observed bio at or before the same cutoff."
        ),
    )
    parser.add_argument("--max-files", type=int, default=0, help="Use only the first N parquet files after sorting.")
    parser.add_argument("--strict-dates", action="store_true", help="Fail if any parquet rows have an invalid date.")
    parser.add_argument("--history-fraction", type=float, default=DEFAULT_HISTORY_FRACTION)
    parser.add_argument(
        "--future-target-mode",
        "--future_target_mode",
        dest="future_target_mode",
        choices=["new_only", "all_future"],
        default="new_only",
        help="How to define temporal LP targets from the future retweet slice.",
    )
    parser.add_argument("--no-temporal-views", action="store_true")
    parser.add_argument("--duckdb-memory-limit", default="")
    parser.add_argument("--duckdb-threads", type=int, default=0)
    parser.add_argument("--duckdb-temp-dir", default="")
    return parser


def _load_yaml_config(path: str) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on environment.
        raise RuntimeError("PyYAML is required to load --config") from exc
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return loaded


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    config_only, _ = parser.parse_known_args()
    if config_only.config:
        known_dests = {action.dest for action in parser._actions}
        config_defaults = {
            key: value
            for key, value in _load_yaml_config(config_only.config).items()
            if key in known_dests
        }
        parser.set_defaults(**config_defaults)
    return parser.parse_args()


def resolve_input_files(parquet_root: str, parquet_paths: list[str], max_files: int) -> list[str]:
    candidates = [Path(path) for path in parquet_paths] if parquet_paths else [Path(parquet_root)]
    files: list[str] = []
    for candidate in candidates:
        if candidate.is_file():
            if candidate.suffix == ".parquet":
                files.append(candidate.as_posix())
            continue
        if candidate.is_dir():
            files.extend(path.as_posix() for path in sorted(candidate.rglob("*.parquet")))
            continue
        raise FileNotFoundError(f"Parquet path does not exist: {candidate}")

    files = sorted(set(files))
    if max_files > 0:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(
            f"No parquet files found from parquet_root={parquet_root!r} parquet_path={parquet_paths!r}"
        )
    return files


def configure_duckdb(conn: Any, args: argparse.Namespace, out_path: Path) -> None:
    temp_dir = Path(args.duckdb_temp_dir) if args.duckdb_temp_dir else out_path.parent / "_duckdb_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    conn.execute(f"SET temp_directory={_sql_literal(temp_dir)}")
    if args.duckdb_memory_limit:
        conn.execute(f"SET memory_limit={_sql_literal(args.duckdb_memory_limit)}")
    if args.duckdb_threads > 0:
        conn.execute(f"SET threads={int(args.duckdb_threads)}")


def _source_columns(conn: Any, parquet_files: list[str]) -> set[str]:
    rows = conn.execute(
        f"DESCRIBE SELECT * FROM read_parquet({_sql_path_list(parquet_files)})"
    ).fetchall()
    return {str(row[0]) for row in rows}


def _varchar_expr(columns: set[str], expr: str) -> str | None:
    if expr in columns:
        return f"CAST({_quote_identifier(expr)} AS VARCHAR)"
    if "." not in expr:
        return None
    top_level, remainder = expr.split(".", 1)
    if top_level not in columns:
        return None
    return f"json_extract_string(to_json({_quote_identifier(top_level)}), '$.{remainder}')"


def _coalesce_try_cast(columns: set[str], candidates: list[str], target_type: str) -> str:
    exprs = []
    for expr in candidates:
        varchar_expr = _varchar_expr(columns, expr)
        if varchar_expr is not None:
            exprs.append(f"try_cast({varchar_expr} AS {target_type})")
    if not exprs:
        return "NULL"
    if len(exprs) == 1:
        return exprs[0]
    return f"COALESCE({', '.join(exprs)})"


def _coalesce_handle(columns: set[str], candidates: list[str]) -> str:
    exprs = []
    for expr in candidates:
        varchar_expr = _varchar_expr(columns, expr)
        if varchar_expr is not None:
            exprs.append(f"NULLIF(lower(trim({varchar_expr})), '')")
    if not exprs:
        return "NULL"
    if len(exprs) == 1:
        return exprs[0]
    return f"COALESCE({', '.join(exprs)})"


def _require_expr(expr: str, candidates: list[str], field_name: str, columns: set[str]) -> str:
    if expr != "NULL":
        return expr
    candidates_text = ", ".join(candidates)
    columns_text = ", ".join(sorted(columns))
    raise ValueError(
        f"Could not find a usable {field_name} column in parquet input. "
        f"Tried: {candidates_text}. Available columns: {columns_text}"
    )


def _timestamp_value_expr(columns: set[str], expr: str) -> str | None:
    varchar_expr = _varchar_expr(columns, expr)
    if varchar_expr is None:
        return None
    numeric_expr = f"try_cast({varchar_expr} AS DOUBLE)"
    return f"""
        COALESCE(
            try_strptime({varchar_expr}, '%a %b %d %H:%M:%S +0000 %Y'),
            try_cast({varchar_expr} AS TIMESTAMP),
            CASE
                WHEN {numeric_expr} >= 1000000000000 THEN CAST(to_timestamp({numeric_expr} / 1000.0) AS TIMESTAMP)
                WHEN {numeric_expr} >= 1000000000 THEN CAST(to_timestamp({numeric_expr}) AS TIMESTAMP)
                ELSE NULL
            END
        )
    """


def _timestamp_expr(columns: set[str]) -> str:
    exprs: list[str] = []
    for expr in TIMESTAMP_CANDIDATES:
        candidate_expr = _timestamp_value_expr(columns, expr)
        if candidate_expr is not None:
            exprs.append(candidate_expr)
    if not exprs:
        return "NULL"
    if len(exprs) == 1:
        return exprs[0]
    return f"COALESCE({', '.join(exprs)})"


def _build_source_scan(conn: Any, parquet_files: list[str]) -> None:
    columns = _source_columns(conn, parquet_files)
    userid_candidates = [
        "userid",
        "user_id",
        "user_id_str",
        "uid",
        "author_id",
        "from_user_id",
        "from_user_id_str",
        "screen_userid",
        "user.id",
        "user.id_str",
        "user.id_str_h",
    ]
    rt_userid_candidates = [
        "rt_userid",
        "rt_user_id",
        "rt_user_id_str",
        "retweeted_userid",
        "retweeted_user_id",
        "retweeted_user_id_str",
        "retweet_userid",
        "retweet_user_id",
        "retweeted_author_id",
        "retweeted_status_user_id",
        "retweeted_status.user.id",
        "retweeted_status.user.id_str",
        "retweeted_status.user.id_str_h",
    ]
    userid_expr = _coalesce_try_cast(
        columns,
        userid_candidates,
        "BIGINT",
    )
    rt_userid_expr = _coalesce_try_cast(
        columns,
        rt_userid_candidates,
        "BIGINT",
    )
    rt_screen_expr = _coalesce_handle(
        columns,
        [
            "rt_screen",
            "rt_screen_name",
            "rt_user_screen_name",
            "retweeted_screen_name",
            "retweeted_user_screen_name",
            "retweet_screen_name",
            "retweeted_status_user_screen_name",
            "retweeted_status.user.screen_name",
        ],
    )
    observed_at_expr = _timestamp_expr(columns)
    userid_expr = _require_expr(userid_expr, userid_candidates, "source user id", columns)
    rt_userid_expr = _require_expr(rt_userid_expr, rt_userid_candidates, "retweeted user id", columns)
    observed_at_expr = _require_expr(
        observed_at_expr,
        TIMESTAMP_CANDIDATES,
        "timestamp",
        columns,
    )

    conn.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW source_scan AS
        SELECT
            {userid_expr} AS userid,
            {rt_userid_expr} AS rt_userid,
            {rt_screen_expr} AS rt_screen,
            {observed_at_expr} AS observed_at
        FROM read_parquet({_sql_path_list(parquet_files)})
        """
    )


def _validate_dates(conn: Any, strict_dates: bool) -> int:
    invalid_dates = int(
        conn.execute(
            """
            SELECT count(*)
            FROM source_scan
            WHERE observed_at IS NULL
            """
        ).fetchone()[0]
    )
    if invalid_dates and strict_dates:
        raise ValueError(f"Encountered {invalid_dates:,} parquet rows with invalid dates.")
    return invalid_dates


def _cutoff_filter_sql(cutoff: str) -> str:
    if not cutoff:
        return ""
    return f"AND observed_at <= CAST({_sql_literal(cutoff)} AS TIMESTAMP)"


def _build_retweet_events(conn: Any, graph_cutoff: str) -> tuple[int, int]:
    conn.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE retweet_events AS
        SELECT
            userid,
            rt_userid,
            rt_screen,
            observed_at
        FROM source_scan
        WHERE userid IS NOT NULL
            AND rt_userid IS NOT NULL
            AND userid <> rt_userid
            AND observed_at IS NOT NULL
            {_cutoff_filter_sql(graph_cutoff)}
        """
    )
    counts = conn.execute(
        """
        SELECT count(*) AS rows, count(DISTINCT userid) + count(DISTINCT rt_userid) AS node_count_hint
        FROM retweet_events
        """
    ).fetchone()
    return int(counts[0]), int(counts[1])


def fetch_edge_table(conn: Any) -> pa.Table:
    return conn.execute(
        """
        SELECT
            userid,
            rt_userid,
            count(*)::BIGINT AS n_retweets
        FROM retweet_events
        GROUP BY userid, rt_userid
        ORDER BY userid, rt_userid
        """
    ).fetch_arrow_table()


def build_temporal_views(
    conn: Any,
    u2i: dict[int, int],
    history_fraction: float,
    future_target_mode: str,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, list[str]], dict[str, int]]:
    total_events = int(conn.execute("SELECT count(*) FROM retweet_events").fetchone()[0])
    if total_events < 2:
        hist_table = conn.execute(
            """
            SELECT
                userid,
                rt_userid,
                count(*)::BIGINT AS n_retweets
            FROM retweet_events
            GROUP BY userid, rt_userid
            ORDER BY userid, rt_userid
            """
        ).fetch_arrow_table()
        hist_edge_index, hist_edge_attr = edge_table_to_tensors(hist_table, u2i)
        empty_index = torch.zeros((2, 0), dtype=torch.long)
        return (
            {"retweet_all": None, "temporal_history": hist_edge_index, "future_edge_index": empty_index},
            {"retweet_all": None, "temporal_history": hist_edge_attr},
            {"retweet_all": EDGE_ATTR_FEATURE_NAMES, "temporal_history": EDGE_ATTR_FEATURE_NAMES},
            {
                "history_rows": total_events,
                "future_rows": 0,
                "history_edges": int(hist_edge_index.shape[1]),
                "future_edges": 0,
                "future_overlap_edges": 0,
                "future_target_edges": 0,
            },
        )

    cutoff_idx = int(total_events * history_fraction)
    cutoff_idx = max(1, min(total_events - 1, cutoff_idx))
    conn.execute(
        """
        CREATE OR REPLACE TEMP VIEW ordered_retweet_events AS
        SELECT
            userid,
            rt_userid,
            observed_at,
            row_number() OVER (ORDER BY observed_at, userid, rt_userid) AS row_num
        FROM retweet_events
        """
    )
    hist_table = conn.execute(
        f"""
        SELECT
            userid,
            rt_userid,
            count(*)::BIGINT AS n_retweets
        FROM ordered_retweet_events
        WHERE row_num <= {cutoff_idx}
        GROUP BY userid, rt_userid
        ORDER BY userid, rt_userid
        """
    ).fetch_arrow_table()
    fut_table = conn.execute(
        f"""
        SELECT
            userid,
            rt_userid,
            count(*)::BIGINT AS n_retweets
        FROM ordered_retweet_events
        WHERE row_num > {cutoff_idx}
        GROUP BY userid, rt_userid
        ORDER BY userid, rt_userid
        """
    ).fetch_arrow_table()

    hist_edge_index, hist_edge_attr = edge_table_to_tensors(hist_table, u2i)
    fut_pairs = arrow_pairs(fut_table)
    hist_pairs = arrow_pairs(hist_table)
    target_pairs = fut_pairs - hist_pairs if future_target_mode == "new_only" else fut_pairs
    target_edge_index = build_target_edge_index(target_pairs, u2i)

    stats = {
        "history_rows": cutoff_idx,
        "future_rows": total_events - cutoff_idx,
        "history_edges": int(hist_edge_index.shape[1]),
        "future_edges": int(len(fut_pairs)),
        "future_overlap_edges": int(len(hist_pairs & fut_pairs)),
        "future_target_edges": int(target_edge_index.shape[1]),
    }
    return (
        {"retweet_all": None, "temporal_history": hist_edge_index, "future_edge_index": target_edge_index},
        {"retweet_all": None, "temporal_history": hist_edge_attr},
        {"retweet_all": EDGE_ATTR_FEATURE_NAMES, "temporal_history": EDGE_ATTR_FEATURE_NAMES},
        stats,
    )


def build_user_ids(edge_table: pa.Table) -> list[int]:
    src = edge_table.column("userid").to_numpy(zero_copy_only=False)
    dst = edge_table.column("rt_userid").to_numpy(zero_copy_only=False)
    return np.unique(np.concatenate([src, dst])).astype(np.int64).tolist()


def edge_table_to_tensors(edge_table: pa.Table, u2i: dict[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    src = edge_table.column("userid").to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    dst = edge_table.column("rt_userid").to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    weight = edge_table.column("n_retweets").to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
    src_idx = np.fromiter((u2i[int(user_id)] for user_id in src), dtype=np.int64, count=len(src))
    dst_idx = np.fromiter((u2i[int(user_id)] for user_id in dst), dtype=np.int64, count=len(dst))
    edge_index = torch.from_numpy(np.vstack([src_idx, dst_idx])).long()
    edge_attr = torch.from_numpy(weight.reshape(-1, 1)).float()
    return edge_index, edge_attr


def arrow_pairs(edge_table: pa.Table) -> set[tuple[int, int]]:
    src = edge_table.column("userid").to_numpy(zero_copy_only=False)
    dst = edge_table.column("rt_userid").to_numpy(zero_copy_only=False)
    return {(int(u), int(v)) for u, v in zip(src.tolist(), dst.tolist())}


def build_target_edge_index(target_pairs: set[tuple[int, int]], u2i: dict[int, int]) -> torch.Tensor:
    if not target_pairs:
        return torch.zeros((2, 0), dtype=torch.long)
    ordered_pairs = sorted(target_pairs)
    src = np.fromiter((u2i[src_user] for src_user, _ in ordered_pairs), dtype=np.int64, count=len(ordered_pairs))
    dst = np.fromiter((u2i[dst_user] for _, dst_user in ordered_pairs), dtype=np.int64, count=len(ordered_pairs))
    return torch.from_numpy(np.vstack([src, dst])).long()


def fetch_handles(conn: Any, user_ids: list[int]) -> list[str | None]:
    if not user_ids:
        return []
    node_table = pa.table({"userid": pa.array(user_ids, type=pa.int64())})
    conn.register("graph_nodes", node_table)
    try:
        handle_table = conn.execute(
            """
            WITH handle_events AS (
                SELECT rt_userid AS userid, lower(rt_screen) AS handle, observed_at
                FROM retweet_events
                WHERE rt_screen IS NOT NULL
            ),
            ranked AS (
                SELECT
                    h.userid,
                    h.handle,
                    row_number() OVER (
                        PARTITION BY h.userid
                        ORDER BY h.observed_at DESC, h.handle DESC
                    ) AS rn
                FROM handle_events AS h
                INNER JOIN graph_nodes AS g
                    ON h.userid = g.userid
                WHERE h.handle IS NOT NULL AND trim(h.handle) <> ''
            )
            SELECT userid, handle
            FROM ranked
            WHERE rn = 1
            ORDER BY userid
            """
        ).fetch_arrow_table()
    finally:
        conn.unregister("graph_nodes")

    handle_map = {
        int(user_id): str(handle)
        for user_id, handle in zip(
            handle_table.column("userid").to_pylist(),
            handle_table.column("handle").to_pylist(),
        )
    }
    return [handle_map.get(int(user_id)) for user_id in user_ids]


def resolve_bio_embeddings(
    conn: Any,
    user_ids: list[int],
    bio_embeddings_root: Path,
    graph_cutoff: str,
) -> tuple[torch.Tensor, list[str], dict[str, Any]]:
    phase_started = time.monotonic()
    node_table = pa.table({"userid": pa.array(user_ids, type=pa.int64())})
    conn.register("graph_nodes", node_table)
    user_bio_path = bio_embeddings_root / "user_bio_observations.parquet"
    bio_index_path = bio_embeddings_root / "bio_embedding_index.parquet"
    if not user_bio_path.exists():
        raise FileNotFoundError(f"Missing bio observations parquet: {user_bio_path}")
    if not bio_index_path.exists():
        raise FileNotFoundError(f"Missing bio embedding index parquet: {bio_index_path}")

    cutoff_filter = ""
    effective_last_seen = "COALESCE(last_seen_at, first_seen_at)"
    if graph_cutoff:
        cutoff_sql = f"CAST({_sql_literal(graph_cutoff)} AS TIMESTAMP)"
        cutoff_filter = f"AND COALESCE(first_seen_at, last_seen_at) <= {cutoff_sql}"
        effective_last_seen = f"LEAST(COALESCE(last_seen_at, first_seen_at), {cutoff_sql})"

    try:
        selected = conn.execute(
            f"""
            WITH candidate_bios AS (
                SELECT
                    CAST(userid AS BIGINT) AS userid,
                    CAST(bio_hash AS VARCHAR) AS bio_hash,
                    try_cast(first_seen_at AS TIMESTAMP) AS first_seen_at,
                    try_cast(last_seen_at AS TIMESTAMP) AS last_seen_at
                FROM read_parquet({_sql_literal(user_bio_path)})
                WHERE CAST(userid AS BIGINT) IN (SELECT userid FROM graph_nodes)
                    AND bio_hash IS NOT NULL
                    AND trim(CAST(bio_hash AS VARCHAR)) <> ''
                    {cutoff_filter}
            ),
            ranked AS (
                SELECT
                    userid,
                    bio_hash,
                    row_number() OVER (
                        PARTITION BY userid
                        ORDER BY {effective_last_seen} DESC NULLS LAST, bio_hash DESC
                    ) AS rn
                FROM candidate_bios
            )
            SELECT
                r.userid,
                r.bio_hash,
                i.embedding_shard,
                i.embedding_row,
                i.embedding_dim
            FROM ranked AS r
            LEFT JOIN read_parquet({_sql_literal(bio_index_path)}) AS i
                ON r.bio_hash = i.bio_hash
            WHERE r.rn = 1
            ORDER BY r.userid
            """
        ).fetch_arrow_table()
    finally:
        conn.unregister("graph_nodes")

    embedding_dims = [
        int(value)
        for value in selected.column("embedding_dim").to_pylist()
        if value is not None
    ] if selected.num_rows > 0 else []
    if not embedding_dims:
        embedding_dim_row = conn.execute(
            f"""
            SELECT max(CAST(embedding_dim AS BIGINT))
            FROM read_parquet({_sql_literal(bio_index_path)})
            """
        ).fetchone()
        if embedding_dim_row and embedding_dim_row[0] is not None:
            embedding_dims = [int(embedding_dim_row[0])]
    if not embedding_dims:
        raise RuntimeError("Matched bios but could not resolve any embedding dimensions from bio_embedding_index.")
    embedding_dim = max(embedding_dims)
    features = np.zeros((len(user_ids), embedding_dim), dtype=np.float32)
    u2i = {int(user_id): idx for idx, user_id in enumerate(user_ids)}

    shard_to_nodes: dict[str, list[tuple[int, int]]] = {}
    missing_bio = 0
    for user_id, shard_path, shard_row in zip(
        selected.column("userid").to_pylist(),
        selected.column("embedding_shard").to_pylist(),
        selected.column("embedding_row").to_pylist(),
    ):
        if shard_path is None or shard_row is None:
            missing_bio += 1
            continue
        shard_to_nodes.setdefault(str(shard_path), []).append((u2i[int(user_id)], int(shard_row)))

    _log_progress(
        "resolved bio selections for "
        f"{selected.num_rows:,} users; loading {len(shard_to_nodes):,} embedding shard(s)"
    )
    matched_users = 0
    total_shards = len(shard_to_nodes)
    for shard_idx, (shard_path, entries) in enumerate(sorted(shard_to_nodes.items()), start=1):
        _log_progress(
            f"loading bio shard {shard_idx:,}/{total_shards:,} "
            f"for {len(entries):,} user(s)"
        )
        shard_abs = Path(shard_path)
        if not shard_abs.is_absolute():
            shard_abs = bio_embeddings_root / shard_abs
        shard_vectors = np.load(shard_abs, mmap_mode="r")
        node_rows = np.fromiter((node_idx for node_idx, _ in entries), dtype=np.int64, count=len(entries))
        shard_rows = np.fromiter((row_idx for _, row_idx in entries), dtype=np.int64, count=len(entries))
        features[node_rows] = np.asarray(shard_vectors[shard_rows], dtype=np.float32)
        matched_users += len(entries)

    feature_names = [f"bio_emb_{idx}" for idx in range(embedding_dim)]
    stats = {
        "policy": "latest_observed_bio_at_or_before_cutoff" if graph_cutoff else "latest_observed_bio_overall",
        "embedding_dim": embedding_dim,
        "matched_users": matched_users,
        "missing_users": len(user_ids) - matched_users,
        "selected_users": int(selected.num_rows),
        "missing_embedding_rows": missing_bio,
    }
    _log_progress(
        "finished bio feature resolution in "
        f"{_format_elapsed(time.monotonic() - phase_started)} "
        f"(matched={matched_users:,}, missing={len(user_ids) - matched_users:,})"
    )
    return torch.from_numpy(features).float(), feature_names, stats


def validate_graph_artifact(graph_obj: dict[str, Any]) -> None:
    data = graph_obj["data"]
    user_ids = graph_obj["user_ids"]
    feature_names = graph_obj["feature_names"]
    edge_attr_feature_names = graph_obj["edge_attr_feature_names"]

    if data.x.dim() != 2:
        raise ValueError("data.x must be 2D")
    if data.edge_index.dim() != 2 or data.edge_index.shape[0] != 2:
        raise ValueError("data.edge_index must have shape [2, E]")
    if len(user_ids) != data.x.shape[0]:
        raise ValueError("len(user_ids) must equal data.x.shape[0]")
    if len(feature_names) != data.x.shape[1]:
        raise ValueError("len(feature_names) must equal data.x.shape[1]")
    if data.edge_attr is not None:
        if data.edge_attr.dim() != 2:
            raise ValueError("data.edge_attr must be 2D when present")
        if data.edge_attr.shape[0] != data.edge_index.shape[1]:
            raise ValueError("data.edge_attr rows must equal edge count")
        if len(edge_attr_feature_names) != data.edge_attr.shape[1]:
            raise ValueError("edge_attr_feature_names must align with data.edge_attr")
    if torch.isnan(data.x).any():
        raise ValueError("data.x contains NaN")


def main() -> None:
    started = time.monotonic()
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    parquet_files = resolve_input_files(args.parquet_root, args.parquet_path, args.max_files)
    print("Configuration")
    print(f"  parquet_files: {len(parquet_files):,}")
    print(f"  out: {out_path}")
    print(f"  bio_embeddings_root: {args.bio_embeddings_root}")
    print(f"  graph_cutoff: {args.graph_cutoff or '<none>'}")
    print(f"  history_fraction: {args.history_fraction}")
    print(f"  future_target_mode: {args.future_target_mode}")
    print(f"  temporal_views: {not args.no_temporal_views}")
    print()

    _log_progress(f"resolved {len(parquet_files):,} parquet file(s)")
    duckdb = _require_duckdb()
    conn = duckdb.connect()
    configure_duckdb(conn, args, out_path)

    try:
        phase_started = time.monotonic()
        _log_progress("building DuckDB source scan over parquet input")
        _build_source_scan(conn, parquet_files)
        _log_progress(f"source scan ready in {_format_elapsed(time.monotonic() - phase_started)}")

        phase_started = time.monotonic()
        _log_progress("checking date parse coverage")
        invalid_dates = _validate_dates(conn, args.strict_dates)
        _log_progress(
            "date validation complete in "
            f"{_format_elapsed(time.monotonic() - phase_started)}; "
            f"invalid rows dropped={invalid_dates:,}"
        )

        phase_started = time.monotonic()
        _log_progress("materializing cleaned retweet events")
        retweet_rows, _ = _build_retweet_events(conn, args.graph_cutoff)
        if retweet_rows == 0:
            raise RuntimeError("No valid retweet events found after parquet cleaning.")
        _log_progress(
            f"retweet events ready in {_format_elapsed(time.monotonic() - phase_started)} "
            f"with {retweet_rows:,} rows"
        )

        phase_started = time.monotonic()
        _log_progress("aggregating directed retweet edges")
        edge_table = fetch_edge_table(conn)
        _log_progress(
            f"edge aggregation complete in {_format_elapsed(time.monotonic() - phase_started)} "
            f"with {edge_table.num_rows:,} unique edges"
        )
        user_ids = build_user_ids(edge_table)
        u2i = {int(user_id): idx for idx, user_id in enumerate(user_ids)}
        edge_index, edge_attr = edge_table_to_tensors(edge_table, u2i)
        _log_progress(f"constructed stable user index with {len(user_ids):,} node(s)")

        phase_started = time.monotonic()
        _log_progress("resolving latest available handles for graph nodes")
        handles = fetch_handles(conn, user_ids)
        _log_progress(f"handle resolution complete in {_format_elapsed(time.monotonic() - phase_started)}")

        x, feature_names, bio_stats = resolve_bio_embeddings(
            conn=conn,
            user_ids=user_ids,
            bio_embeddings_root=Path(args.bio_embeddings_root),
            graph_cutoff=args.graph_cutoff,
        )

        temporal_stats: dict[str, Any] = {}
        edge_index_views: dict[str, torch.Tensor] = {}
        edge_attr_views: dict[str, torch.Tensor] = {}
        edge_attr_feature_names_views: dict[str, list[str]] = {}
        target_edge_index_views: dict[str, torch.Tensor] = {}
        if not args.no_temporal_views:
            phase_started = time.monotonic()
            _log_progress("building temporal edge views")
            view_index, view_attr, view_names, temporal_stats = build_temporal_views(
                conn=conn,
                u2i=u2i,
                history_fraction=args.history_fraction,
                future_target_mode=args.future_target_mode,
            )
            _log_progress(
                "temporal views ready in "
                f"{_format_elapsed(time.monotonic() - phase_started)} "
                f"(history_edges={temporal_stats['history_edges']:,}, "
                f"future_targets={temporal_stats['future_target_edges']:,})"
            )
            edge_index_views = {
                "retweet_all": edge_index,
                "temporal_history": view_index["temporal_history"],
            }
            edge_attr_views = {
                "retweet_all": edge_attr,
                "temporal_history": view_attr["temporal_history"],
            }
            edge_attr_feature_names_views = {
                "retweet_all": EDGE_ATTR_FEATURE_NAMES,
                "temporal_history": view_names["temporal_history"],
            }
            target_edge_index_views = {
                "temporal_new": view_index["future_edge_index"],
            }

        y = torch.full((len(user_ids),), -1, dtype=torch.long)
        data = _make_data_object(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        data.feature_names = feature_names
        data.edge_attr_feature_names = EDGE_ATTR_FEATURE_NAMES
        data.user_ids = list(user_ids)

        graph_obj: dict[str, Any] = {
            "x": x,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "edge_attr_feature_names": EDGE_ATTR_FEATURE_NAMES,
            "user_ids": user_ids,
            "u2i": u2i,
            "feature_names": feature_names,
            "handles": handles,
            "y": y,
            "label_names": [],
            "bio_embedding_policy": bio_stats["policy"],
            "data": data,
        }
        if edge_index_views:
            graph_obj["edge_index_views"] = edge_index_views
            graph_obj["edge_attr_views"] = edge_attr_views
            graph_obj["edge_attr_feature_names_views"] = edge_attr_feature_names_views
            graph_obj["target_edge_index_views"] = target_edge_index_views
            graph_obj["future_edge_index"] = target_edge_index_views["temporal_new"]

        phase_started = time.monotonic()
        _log_progress("validating graph artifact in memory")
        validate_graph_artifact(graph_obj)
        _log_progress(f"artifact validation complete in {_format_elapsed(time.monotonic() - phase_started)}")

        # Benchmark targets: node-regression profile panel + static-LP edge views,
        # so future builds emit them by default. Opt out with SKIP_BENCHMARK_TARGETS=1.
        try:
            import os

            if os.environ.get("SKIP_BENCHMARK_TARGETS", "") not in {"1", "true", "True"}:
                from enrich_graph_targets import enrich_graph_obj, parquet_scan_sql

                bt_stats = enrich_graph_obj(
                    graph_obj,
                    "midterm",
                    conn=conn,
                    scan_sql=parquet_scan_sql(parquet_files),
                )
                _log_progress(f"benchmark targets attached: static={bt_stats['static_split']}")
        except Exception as exc:  # noqa: BLE001 - enrichment must not break a build
            _log_progress(f"[warn] benchmark-target enrichment failed: {exc}")

        phase_started = time.monotonic()
        _log_progress(f"writing graph artifact to {out_path}")
        torch.save(graph_obj, out_path)
        _log_progress(f"graph artifact written in {_format_elapsed(time.monotonic() - phase_started)}")

        meta = {
            "parquet_root": args.parquet_root,
            "parquet_paths": args.parquet_path,
            "parquet_files_used": len(parquet_files),
            "graph_cutoff": args.graph_cutoff or None,
            "nodes": int(len(user_ids)),
            "edges": int(edge_index.shape[1]),
            "retweet_events": int(retweet_rows),
            "invalid_date_rows_dropped": int(invalid_dates),
            "node_feature_dim": int(x.shape[1]),
            "edge_feature_names": EDGE_ATTR_FEATURE_NAMES,
            "bio_embeddings_root": args.bio_embeddings_root,
            "bio_embedding_policy": bio_stats["policy"],
            "bio_embedding_dim": int(bio_stats["embedding_dim"]),
            "bio_embedding_matched_users": int(bio_stats["matched_users"]),
            "bio_embedding_missing_users": int(bio_stats["missing_users"]),
            "bio_embedding_missing_rows": int(bio_stats["missing_embedding_rows"]),
            "history_fraction": args.history_fraction,
            "future_target_mode": args.future_target_mode,
            "temporal": temporal_stats,
        }
        meta_path = out_path.with_suffix(".meta.json")
        phase_started = time.monotonic()
        _log_progress(f"writing metadata to {meta_path}")
        with meta_path.open("w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2)
        _log_progress(
            f"metadata written in {_format_elapsed(time.monotonic() - phase_started)}; "
            f"total elapsed={_format_elapsed(time.monotonic() - started)}"
        )

        print(f"Saved graph: {out_path}")
        print(f"Saved meta:  {meta_path}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
