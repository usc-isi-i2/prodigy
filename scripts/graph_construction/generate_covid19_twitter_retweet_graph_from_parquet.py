"""Build a covid19_twitter retweet graph directly from staged parquet files.

This preserves the current covid19_twitter graph contract while swapping the
raw JSON ingestion step for a parquet-backed loader.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_PARQUET_ROOT = "/dataMeR2/phil/data/covid19_twitter/parquet"
DEFAULT_BIO_EMBEDDINGS_ROOT = "/dataMeR2/phil/data/covid19_twitter/bio_embeddings/gte-multilingual-base/version=v001"
DEFAULT_OUT = "data/data/covid19_twitter/graphs/retweet_graph_parquet.pt"
DEFAULT_HISTORY_FRACTION = 0.3
DEFAULT_LABELS_PARQUET_GLOB = ""


def _log_progress(message: str) -> None:
    print(f"[progress] {message}", flush=True)


def _format_elapsed(seconds: float) -> str:
    return f"{seconds:.1f}s"


def _require_duckdb() -> Any:
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - depends on environment.
        raise RuntimeError(
            "duckdb is required to build the parquet covid19_twitter graph. "
            "Install the dataset environment with duckdb, pandas, pyarrow, sklearn, and torch."
        ) from exc
    return duckdb


def _load_covid_graph_module() -> Any:
    module_path = REPO_ROOT / "data" / "data" / "covid19_twitter" / "scripts" / "generate_user_graph.py"
    spec = importlib.util.spec_from_file_location("covid19_twitter_generate_user_graph", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load covid19_twitter helpers from {module_path}")

    def _exec_module() -> Any:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    try:
        return _exec_module()
    except ModuleNotFoundError as exc:
        if exc.name != "torch_geometric":
            raise

    fake_tg = ModuleType("torch_geometric")
    fake_tg_data = ModuleType("torch_geometric.data")

    class _FallbackData(SimpleNamespace):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)

    fake_tg_data.Data = _FallbackData
    fake_tg.data = fake_tg_data
    prev_tg = sys.modules.get("torch_geometric")
    prev_tg_data = sys.modules.get("torch_geometric.data")
    sys.modules["torch_geometric"] = fake_tg
    sys.modules["torch_geometric.data"] = fake_tg_data
    try:
        return _exec_module()
    finally:
        if prev_tg is None:
            sys.modules.pop("torch_geometric", None)
        else:
            sys.modules["torch_geometric"] = prev_tg
        if prev_tg_data is None:
            sys.modules.pop("torch_geometric.data", None)
        else:
            sys.modules["torch_geometric.data"] = prev_tg_data


COVID_GRAPH = _load_covid_graph_module()
Data = COVID_GRAPH.Data
EDGE_ATTR_FEATURE_NAMES = ["n_retweets"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a directed covid19_twitter retweet graph from staged parquet input "
            "while preserving the existing covid graph schema."
        )
    )
    parser.add_argument("--parquet-root", default="/dataMeR2/phil/data/covid19_twitter/parquet")
    parser.add_argument(
        "--parquet-path",
        action="append",
        default=[],
        help="Optional parquet file or directory path. Repeatable. Defaults to --parquet-root recursively.",
    )
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument(
        "--bio-embeddings-root",
        default=DEFAULT_BIO_EMBEDDINGS_ROOT,
        help="Bio embedding store root matching the ukr-rus parquet graph flow.",
    )
    parser.add_argument("--embeddings", default="", help="Optional user_embeddings_*.pt artifact.")
    parser.add_argument("--embedding-pool", choices=["meanpool", "maxpool"], default="meanpool")
    parser.add_argument(
        "--labels-parquet-glob",
        default=DEFAULT_LABELS_PARQUET_GLOB,
        help="Optional external node label parquet glob keyed by screen_name.",
    )
    parser.add_argument(
        "--graph-cutoff",
        default="",
        help="Optional inclusive cutoff timestamp applied before graph and temporal-view construction.",
    )
    parser.add_argument("--max-files", type=int, default=0, help="Use only the first N parquet files after sorting.")
    parser.add_argument("--strict-dates", action="store_true", help="Fail if any parquet rows have an invalid date.")
    parser.add_argument("--history-fraction", type=float, default=DEFAULT_HISTORY_FRACTION)
    parser.add_argument(
        "--future-target-mode",
        choices=["new_only", "all_future"],
        default="new_only",
        help="How to define temporal LP targets from the future retweet slice.",
    )
    parser.add_argument("--no-temporal-views", action="store_true")
    parser.add_argument(
        "--keep-isolates",
        dest="keep_isolates",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--duckdb-memory-limit", default="")
    parser.add_argument("--duckdb-threads", type=int, default=0)
    parser.add_argument("--duckdb-temp-dir", default="")
    return parser.parse_args()


def _sql_literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _sql_path_list(paths: list[str]) -> str:
    return "[" + ", ".join(_sql_literal(path) for path in paths) + "]"


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


def _column_exists(columns: set[str], expr: str) -> bool:
    return expr.split(".", 1)[0] in columns


def _varchar_expr(columns: set[str], expr: str) -> str | None:
    if not _column_exists(columns, expr):
        return None
    if "." not in expr:
        return f"CAST({expr} AS VARCHAR)"
    top_level, remainder = expr.split(".", 1)
    return f"json_extract_string(to_json({top_level}), '$.{remainder}')"


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


def _timestamp_expr(columns: set[str]) -> str:
    exprs: list[str] = []
    if "observed_at" in columns:
        exprs.append("try_cast(observed_at AS TIMESTAMP)")
    if "created_ts" in columns:
        exprs.append("try_cast(created_ts AS TIMESTAMP)")
    if "created_at" in columns:
        exprs.append("try_strptime(CAST(created_at AS VARCHAR), '%a %b %d %H:%M:%S +0000 %Y')")
    if "date" in columns:
        exprs.append("try_strptime(CAST(date AS VARCHAR), '%a %b %d %H:%M:%S +0000 %Y')")
        exprs.append("try_cast(date AS TIMESTAMP)")
    if not exprs:
        return "NULL"
    if len(exprs) == 1:
        return exprs[0]
    return f"COALESCE({', '.join(exprs)})"


def _first_available_expr(columns: set[str], candidates: list[str], default: str = "NULL") -> str:
    for expr in candidates:
        if _column_exists(columns, expr):
            return expr
    return default


def _make_data_object(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    y: torch.Tensor,
) -> Any:
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def resolve_bio_embeddings(
    conn: Any,
    user_ids: list[int],
    bio_embeddings_root: Path,
    graph_cutoff: str,
) -> tuple[torch.Tensor, list[str], dict[str, Any]]:
    phase_started = time.monotonic()
    user_bio_path = bio_embeddings_root / "user_bio_observations.parquet"
    bio_index_path = bio_embeddings_root / "bio_embedding_index.parquet"
    if not user_bio_path.exists():
        raise FileNotFoundError(f"Missing bio observations parquet: {user_bio_path}")
    if not bio_index_path.exists():
        raise FileNotFoundError(f"Missing bio embedding index parquet: {bio_index_path}")

    conn.register("graph_nodes", pd.DataFrame({"userid": user_ids}))
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
        ).fetchdf()
    finally:
        conn.unregister("graph_nodes")

    embedding_dims = [int(value) for value in selected["embedding_dim"].tolist() if pd.notna(value)] if not selected.empty else []
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
        raise RuntimeError("Could not resolve embedding dimensions from bio_embedding_index.")

    embedding_dim = max(embedding_dims)
    features = np.zeros((len(user_ids), embedding_dim), dtype=np.float32)
    u2i = {int(user_id): idx for idx, user_id in enumerate(user_ids)}

    shard_to_nodes: dict[str, list[tuple[int, int]]] = {}
    missing_bio = 0
    for row in selected.itertuples(index=False):
        if pd.isna(row.embedding_shard) or pd.isna(row.embedding_row):
            missing_bio += 1
            continue
        shard_to_nodes.setdefault(str(row.embedding_shard), []).append((u2i[int(row.userid)], int(row.embedding_row)))

    _log_progress(
        "resolved bio selections for "
        f"{len(selected):,} users; loading {len(shard_to_nodes):,} embedding shard(s)"
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
        "matched_users": int(matched_users),
        "missing_users": int(len(user_ids) - matched_users),
        "selected_users": int(len(selected)),
        "missing_embedding_rows": int(missing_bio),
        "embedding_dim": int(embedding_dim),
    }
    _log_progress(
        "finished bio feature resolution in "
        f"{_format_elapsed(time.monotonic() - phase_started)} "
        f"(matched={matched_users:,}, missing={len(user_ids) - matched_users:,})"
    )
    return torch.from_numpy(features).float(), feature_names, stats


def _build_source_scan(conn: Any, parquet_files: list[str]) -> None:
    columns = _source_columns(conn, parquet_files)
    userid_expr = _coalesce_try_cast(columns, ["userid", "user.id", "user_id"], "BIGINT")
    screen_name_expr = _coalesce_handle(columns, ["screen_name", "user.screen_name"])
    rt_userid_expr = _coalesce_try_cast(
        columns,
        ["rt_userid", "retweeted_status.user.id", "retweeted_user_id"],
        "BIGINT",
    )
    rt_screen_expr = _coalesce_handle(
        columns,
        ["rt_screen", "retweeted_status.user.screen_name", "retweeted_screen_name"],
    )
    followers_count_expr = _coalesce_try_cast(columns, ["followers_count", "user.followers_count"], "DOUBLE")
    statuses_count_expr = _coalesce_try_cast(columns, ["statuses_count", "user.statuses_count"], "DOUBLE")
    verified_expr = _coalesce_try_cast(columns, ["verified", "user.verified"], "BOOLEAN")
    rt_fav_count_expr = _coalesce_try_cast(columns, ["rt_fav_count", "retweeted_status.favorite_count"], "DOUBLE")
    rt_reply_count_expr = _coalesce_try_cast(columns, ["rt_reply_count"], "DOUBLE")
    hashtag_expr = _first_available_expr(columns, ["hashtag", "hashtags", "entities.hashtags"])
    mentions_expr = _first_available_expr(columns, ["mentionsn", "user_mentions", "entities.user_mentions"])
    media_expr = _first_available_expr(columns, ["media_urls", "extended_entities.media"])
    observed_at_expr = _timestamp_expr(columns)

    conn.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW source_scan AS
        SELECT
            {userid_expr} AS userid,
            {screen_name_expr} AS screen_name,
            {rt_userid_expr} AS rt_userid,
            {rt_screen_expr} AS rt_screen,
            {observed_at_expr} AS observed_at,
            {followers_count_expr} AS followers_count,
            {statuses_count_expr} AS statuses_count,
            COALESCE(try_cast({verified_expr} AS INTEGER), 0) AS verified,
            {rt_fav_count_expr} AS rt_fav_count,
            {rt_reply_count_expr} AS rt_reply_count,
            CAST(0.0 AS DOUBLE) AS sent_vader,
            {hashtag_expr} AS hashtag,
            {mentions_expr} AS mentionsn,
            {media_expr} AS media_urls
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


def _coerce_list_value(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        converted = value.tolist()
        if isinstance(converted, list):
            return converted
    return []


def load_raw_rows(conn: Any, graph_cutoff: str) -> pd.DataFrame:
    raw = conn.execute(
        f"""
        SELECT
            userid,
            screen_name,
            rt_userid,
            rt_screen,
            observed_at AS timestamp,
            followers_count,
            verified,
            statuses_count,
            rt_fav_count,
            rt_reply_count,
            sent_vader,
            hashtag,
            mentionsn,
            media_urls
        FROM source_scan
        WHERE observed_at IS NOT NULL
            {_cutoff_filter_sql(graph_cutoff)}
        ORDER BY timestamp, userid
        """
    ).fetchdf()
    for column in ["hashtag", "mentionsn", "media_urls"]:
        if column not in raw.columns:
            raw[column] = [[] for _ in range(len(raw))]
        else:
            raw[column] = raw[column].map(_coerce_list_value)
    return raw


def build_retweet_events(raw: pd.DataFrame) -> pd.DataFrame:
    events = raw.dropna(subset=["userid", "rt_userid"]).copy()
    events = events[events["userid"] != events["rt_userid"]].copy()
    if events.empty:
        raise RuntimeError("No valid retweet rows after cleaning")
    events["userid"] = events["userid"].astype(np.int64)
    events["target_userid"] = events["rt_userid"].astype(np.int64)
    events["target_screen"] = events["rt_screen"]
    return events[
        [
            "userid",
            "screen_name",
            "timestamp",
            "followers_count",
            "verified",
            "statuses_count",
            "rt_fav_count",
            "rt_reply_count",
            "sent_vader",
            "hashtag",
            "mentionsn",
            "media_urls",
            "target_userid",
            "target_screen",
        ]
    ]


def build_temporal_views(
    events: pd.DataFrame,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    u2i: dict[int, int],
    history_fraction: float,
    future_target_mode: str,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, list[str]], dict[str, int]]:
    if len(events) < 2:
        empty_index = torch.zeros((2, 0), dtype=torch.long)
        return (
            {"retweet_all": edge_index, "temporal_history": edge_index},
            {"retweet_all": edge_attr, "temporal_history": edge_attr},
            {"retweet_all": EDGE_ATTR_FEATURE_NAMES, "temporal_history": EDGE_ATTR_FEATURE_NAMES},
            {
                "history_rows": int(len(events)),
                "future_rows": 0,
                "history_edges": int(edge_index.shape[1]),
                "future_edges": 0,
                "future_overlap_edges": 0,
                "future_target_edges": 0,
                "target_edge_index": empty_index,
            },
        )

    events_sorted = events.sort_values("timestamp").reset_index(drop=True)
    cutoff_idx = int(len(events_sorted) * history_fraction)
    cutoff_idx = max(1, min(len(events_sorted) - 1, cutoff_idx))
    hist_events = events_sorted.iloc[:cutoff_idx].copy()
    fut_events = events_sorted.iloc[cutoff_idx:].copy()

    hist_edges_df = (
        hist_events.groupby(["userid", "target_userid"], as_index=False)
        .size()
        .rename(columns={"size": "n_retweets"})
        .sort_values(["userid", "target_userid"], kind="stable")
    )
    fut_edges_df = (
        fut_events.groupby(["userid", "target_userid"], as_index=False)
        .size()
        .rename(columns={"size": "n_retweets"})
        .sort_values(["userid", "target_userid"], kind="stable")
    )
    src = hist_edges_df["userid"].to_numpy(dtype=np.int64, copy=False)
    dst = hist_edges_df["target_userid"].to_numpy(dtype=np.int64, copy=False)
    weight = hist_edges_df["n_retweets"].to_numpy(dtype=np.float32, copy=False)
    src_idx = np.fromiter((u2i[int(user_id)] for user_id in src), dtype=np.int64, count=len(src))
    dst_idx = np.fromiter((u2i[int(user_id)] for user_id in dst), dtype=np.int64, count=len(dst))
    hist_edge_index = torch.from_numpy(np.vstack([src_idx, dst_idx])).long()
    hist_edge_attr = torch.from_numpy(weight.reshape(-1, 1)).float()
    hist_pairs = set(zip(hist_edges_df["userid"], hist_edges_df["target_userid"]))
    fut_pairs = set(zip(fut_edges_df["userid"], fut_edges_df["target_userid"]))
    target_pairs = fut_pairs - hist_pairs if future_target_mode == "new_only" else fut_pairs

    if target_pairs:
        target_df = pd.DataFrame(sorted(target_pairs), columns=["userid", "target_userid"])
        target_df["src"] = target_df["userid"].map(u2i)
        target_df["dst"] = target_df["target_userid"].map(u2i)
        target_df = target_df.dropna(subset=["src", "dst"])
        target_edge_index = torch.tensor(target_df[["src", "dst"]].astype(int).values.T, dtype=torch.long)
    else:
        target_edge_index = torch.zeros((2, 0), dtype=torch.long)

    return (
        {"retweet_all": edge_index, "temporal_history": hist_edge_index},
        {"retweet_all": edge_attr, "temporal_history": hist_edge_attr},
        {"retweet_all": EDGE_ATTR_FEATURE_NAMES, "temporal_history": EDGE_ATTR_FEATURE_NAMES},
        {
            "history_rows": int(len(hist_events)),
            "future_rows": int(len(fut_events)),
            "history_edges": int(hist_edge_index.shape[1]),
            "future_edges": int(len(fut_pairs)),
            "future_overlap_edges": int(len(hist_pairs & fut_pairs)),
            "future_target_edges": int(target_edge_index.shape[1]),
            "target_edge_index": target_edge_index,
        },
    )


def validate_graph_artifact(graph_obj: dict[str, Any]) -> None:
    x = graph_obj["x"]
    y = graph_obj["y"]
    edge_index = graph_obj["edge_index"]
    edge_attr = graph_obj["edge_attr"]
    user_ids = graph_obj["user_ids"]
    if x.shape[0] != len(user_ids):
        raise ValueError(f"x rows {x.shape[0]:,} do not match user_ids {len(user_ids):,}")
    if y.shape[0] != len(user_ids):
        raise ValueError(f"y rows {y.shape[0]:,} do not match user_ids {len(user_ids):,}")
    if edge_index.shape[0] != 2:
        raise ValueError(f"edge_index first dim must be 2, got {tuple(edge_index.shape)}")
    if edge_attr.shape[0] != edge_index.shape[1]:
        raise ValueError(
            f"edge_attr rows {edge_attr.shape[0]:,} do not match edge_index edges {edge_index.shape[1]:,}"
        )


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Configuration")
    print(f"  parquet_root: {args.parquet_root}")
    print(f"  parquet_paths: {args.parquet_path or '<none>'}")
    print(f"  out: {args.out}")
    print(f"  bio_embeddings_root: {args.bio_embeddings_root or '<none>'}")
    print(f"  embeddings: {args.embeddings or '<none>'}")
    print(f"  embedding_pool: {args.embedding_pool}")
    print(f"  labels_parquet_glob: {args.labels_parquet_glob or '<none>'}")
    print(f"  graph_cutoff: {args.graph_cutoff or '<none>'}")
    print(f"  max_files: {args.max_files if args.max_files > 0 else 'all'}")
    print(f"  history_fraction: {args.history_fraction}")
    print(f"  future_target_mode: {args.future_target_mode}")
    print(f"  keep_isolates: {args.keep_isolates}")
    print()

    started = time.monotonic()
    parquet_files = resolve_input_files(args.parquet_root, args.parquet_path, args.max_files)
    _log_progress(f"discovered {len(parquet_files):,} parquet file(s)")

    duckdb = _require_duckdb()
    conn = duckdb.connect(database=":memory:")
    try:
        configure_duckdb(conn, args, out_path)
        _log_progress("building DuckDB source scan")
        _build_source_scan(conn, parquet_files)

        invalid_dates = _validate_dates(conn, args.strict_dates)
        if invalid_dates:
            _log_progress(f"dropping {invalid_dates:,} row(s) with invalid timestamps")

        _log_progress("loading normalized parquet rows")
        raw = load_raw_rows(conn, args.graph_cutoff)
        if raw.empty:
            raise RuntimeError("No usable parquet rows remained after normalization.")
        print(f"Raw frame: rows={len(raw):,} cols={len(raw.columns):,}", flush=True)

        events = build_retweet_events(raw)
        print(
            f"Cleaned retweet events: rows={len(events):,} "
            f"unique_nodes={len(set(events['userid'].tolist()) | set(events['target_userid'].tolist())):,}",
            flush=True,
        )

        user_ids, u2i = COVID_GRAPH.build_user_index(events)
        handles = COVID_GRAPH.build_user_metadata(events, user_ids)
        print(f"Nodes: {len(user_ids):,}", flush=True)

        edges_df = (
            events.groupby(["userid", "target_userid"], as_index=False)
            .size()
            .rename(columns={"size": "n_retweets"})
            .sort_values(["userid", "target_userid"], kind="stable")
        )
        src = edges_df["userid"].to_numpy(dtype=np.int64, copy=False)
        dst = edges_df["target_userid"].to_numpy(dtype=np.int64, copy=False)
        weight = edges_df["n_retweets"].to_numpy(dtype=np.float32, copy=False)
        src_idx = np.fromiter((u2i[int(user_id)] for user_id in src), dtype=np.int64, count=len(src))
        dst_idx = np.fromiter((u2i[int(user_id)] for user_id in dst), dtype=np.int64, count=len(dst))
        edge_index = torch.from_numpy(np.vstack([src_idx, dst_idx])).long()
        edge_attr = torch.from_numpy(weight.reshape(-1, 1)).float()
        print(f"Directed edges: {edge_index.shape[1]:,}", flush=True)

        requested_bio_root = args.bio_embeddings_root.strip()
        bio_root = Path(requested_bio_root) if requested_bio_root else None
        bio_root_exists = bio_root is not None and bio_root.exists()
        if bio_root_exists:
            _log_progress("resolving node features from bio embeddings")
            x, feature_names, emb_stats = resolve_bio_embeddings(
                conn=conn,
                user_ids=user_ids,
                bio_embeddings_root=bio_root,
                graph_cutoff=args.graph_cutoff,
            )
            feature_source = "bio_embeddings"
        elif requested_bio_root and requested_bio_root != DEFAULT_BIO_EMBEDDINGS_ROOT:
            raise FileNotFoundError(f"Bio embeddings root does not exist: {bio_root}")
        elif args.embeddings:
            x, feature_names = COVID_GRAPH.build_node_features(raw, u2i, edges_df)
            x, feature_names, emb_stats = COVID_GRAPH.maybe_attach_embeddings(
                x,
                feature_names,
                user_ids,
                handles,
                args.embeddings,
                args.embedding_pool,
            )
            feature_source = "legacy_embeddings"
        else:
            raise FileNotFoundError(
                "Bio embeddings root does not exist and no legacy --embeddings artifact was provided."
            )

        print(
            f"Feature source={feature_source} dims={x.shape[1]} "
            f"matched_users={emb_stats['matched_users']:,} embedding_dim={emb_stats['embedding_dim']}",
            flush=True,
        )
    finally:
        conn.close()

    n_nodes = len(user_ids)
    out_deg_t = torch.bincount(edge_index[0], minlength=n_nodes)
    in_deg_t = torch.bincount(edge_index[1], minlength=n_nodes)
    isolated_before_drop = int(((out_deg_t == 0) & (in_deg_t == 0)).sum().item())

    if not args.keep_isolates:
        x, edge_index, edge_attr, user_ids, handles, u2i, isolated_dropped = COVID_GRAPH.drop_isolates_from_graph(
            x, edge_index, edge_attr, user_ids, handles
        )
        print(f"Dropped isolated nodes: {isolated_dropped:,}", flush=True)
    else:
        isolated_dropped = 0

    if args.labels_parquet_glob:
        label_info = COVID_GRAPH.load_external_labels(args.labels_parquet_glob)
        y, label_names, label_stats = COVID_GRAPH.build_node_labels(handles, label_info)
        if label_info:
            print(
                f"Attached labels: labeled_nodes={label_stats['labeled_nodes']:,} "
                f"label_count={label_stats['label_count']} labels={label_names}",
                flush=True,
            )
    else:
        y = torch.full((len(user_ids),), -1, dtype=torch.long)
        label_names = []
        label_stats = {"label_count": 0, "labeled_nodes": 0}

    graph_obj = {
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "edge_attr_feature_names": EDGE_ATTR_FEATURE_NAMES,
        "user_ids": user_ids,
        "u2i": u2i,
        "feature_names": feature_names,
        "handles": handles,
        "y": y,
        "label_names": label_names,
        "label_type": "classification",
        "bio_embedding_policy": emb_stats.get("policy", ""),
    }

    temporal_stats: dict[str, Any] = {}
    if not args.no_temporal_views:
        _log_progress("building temporal views")
        edge_index_views, edge_attr_views, edge_name_views, temporal_stats = build_temporal_views(
            events=events,
            edge_index=edge_index,
            edge_attr=edge_attr,
            u2i=u2i,
            history_fraction=args.history_fraction,
            future_target_mode=args.future_target_mode,
        )
        target_edge_index = temporal_stats.pop("target_edge_index")
        graph_obj["edge_index_views"] = edge_index_views
        graph_obj["edge_attr_views"] = edge_attr_views
        graph_obj["edge_attr_feature_names_views"] = edge_name_views
        graph_obj["target_edge_index_views"] = {"temporal_new": target_edge_index}
        graph_obj["future_edge_index"] = target_edge_index
        print(
            "Temporal views: "
            f"history_rows={temporal_stats['history_rows']:,} "
            f"future_rows={temporal_stats['future_rows']:,} "
            f"history_edges={temporal_stats['history_edges']:,} "
            f"future_target_edges={temporal_stats['future_target_edges']:,}",
            flush=True,
        )

    data = _make_data_object(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.feature_names = feature_names
    data.edge_attr_feature_names = EDGE_ATTR_FEATURE_NAMES
    data.label_names = label_names
    data.user_ids = list(user_ids)
    graph_obj["data"] = data

    _log_progress("validating graph artifact")
    validate_graph_artifact(graph_obj)
    torch.save(graph_obj, args.out)

    meta = {
        "parquet_root": args.parquet_root,
        "parquet_paths": args.parquet_path,
        "parquet_file_count": len(parquet_files),
        "out": args.out,
        "graph_cutoff": args.graph_cutoff,
        "nodes": int(len(user_ids)),
        "edges": int(edge_index.shape[1]),
        "node_feature_dim": int(x.shape[1]),
        "edge_feature_names": EDGE_ATTR_FEATURE_NAMES,
        "feature_source": feature_source,
        "bio_embeddings_root": args.bio_embeddings_root,
        "embedding_pool": args.embedding_pool,
        "embeddings": args.embeddings,
        "embedding_dim": emb_stats["embedding_dim"],
        "embedding_matched_users": emb_stats["matched_users"],
        "labels_parquet_glob": args.labels_parquet_glob,
        "label_count": int(label_stats["label_count"]),
        "labeled_nodes": int(label_stats["labeled_nodes"]),
        "keep_isolates": args.keep_isolates,
        "isolated_nodes_before_drop": isolated_before_drop,
        "isolated_nodes_dropped": isolated_dropped,
        "invalid_date_rows": int(invalid_dates),
        "temporal": temporal_stats,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    meta_path = args.out.replace(".pt", ".meta.json")
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

    print(f"Saved graph: {args.out}")
    print(f"Saved meta:  {meta_path}")
    print(f"Elapsed: {_format_elapsed(time.monotonic() - started)}")


if __name__ == "__main__":
    main()
