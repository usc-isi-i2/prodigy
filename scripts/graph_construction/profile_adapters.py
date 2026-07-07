"""Per-dataset adapters that fetch raw user-profile fields for node regression.

Each adapter returns ``raw_by_user``: ``{int user_id -> {field: value}}`` where
fields are a subset of ``benchmark_targets.PROFILE_COUNT_FIELDS`` plus an
``account_creation`` timestamp. The result is fed straight into
``benchmark_targets.build_profile_node_targets``.

Adapters are shared by the standalone enrichment script
(``enrich_graph_targets.py``) and by the graph generators (so future builds emit
the targets by default). They only *read* their source, so they are safe to run
read-only for verification.

Feasibility (verified against the raw data on Tucker):
  * midterm / ukr_rus : flat columns (author profile only) -> ``fetch_profiles_flat``
  * covid19           : nested ``user`` / ``retweeted_status.user`` structs
                        -> ``fetch_profiles_covid19``
  * twibot20          : ``node.json`` ``public_metrics`` -> ``fetch_profiles_twibot20``
  * cp_hk             : NO profile metrics available (bios are free text) -> no adapter
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Optional, Sequence

try:
    from .benchmark_targets import PROFILE_COUNT_FIELDS
except ImportError:  # allow running as a plain script (sys.path insert)
    from benchmark_targets import PROFILE_COUNT_FIELDS


def _register_graph_nodes(conn: Any, user_ids: Sequence[int]) -> None:
    import pyarrow as pa

    conn.register("graph_nodes", pa.table({"userid": pa.array([int(u) for u in user_ids], type=pa.int64())}))


def _arrow_to_raw_by_user(table: Any) -> dict[int, dict[str, Any]]:
    cols = {name: table.column(name).to_pylist() for name in table.column_names}
    n = table.num_rows
    out: dict[int, dict[str, Any]] = {}
    for i in range(n):
        uid = cols["userid"][i]
        if uid is None:
            continue
        rec: dict[str, Any] = {}
        for name in PROFILE_COUNT_FIELDS:
            if name in cols and cols[name][i] is not None:
                rec[name] = cols[name][i]
        if "account_creation" in cols and cols["account_creation"][i] is not None:
            rec["account_creation"] = cols["account_creation"][i]
        out[int(uid)] = rec
    return out


def fetch_profiles_flat(
    conn: Any,
    scan_sql: str,
    user_ids: Sequence[int],
) -> dict[int, dict[str, Any]]:
    """midterm / ukr_rus: flat author-profile columns, latest value per user by ``date``.

    ``scan_sql`` is a FROM-able expression, e.g. ``read_parquet('/…/*.parquet')``
    or a registered view name. Only nodes in ``user_ids`` are returned.
    """
    _register_graph_nodes(conn, user_ids)
    try:
        table = conn.execute(
            f"""
            WITH src AS (
                SELECT
                    CAST(userid AS BIGINT) AS userid,
                    TRY_CAST(followers_count  AS DOUBLE) AS followers_count,
                    TRY_CAST(friends_count    AS DOUBLE) AS friends_count,
                    TRY_CAST(statuses_count   AS DOUBLE) AS statuses_count,
                    TRY_CAST(favourites_count AS DOUBLE) AS favourites_count,
                    TRY_CAST(listed_count     AS DOUBLE) AS listed_count,
                    CAST(account_creation_date AS VARCHAR) AS account_creation,
                    CAST(date AS VARCHAR) AS ts
                FROM {scan_sql}
                WHERE userid IS NOT NULL
            )
            SELECT
                s.userid,
                arg_max(s.followers_count,  s.ts) AS followers_count,
                arg_max(s.friends_count,    s.ts) AS friends_count,
                arg_max(s.statuses_count,   s.ts) AS statuses_count,
                arg_max(s.favourites_count, s.ts) AS favourites_count,
                arg_max(s.listed_count,     s.ts) AS listed_count,
                arg_max(s.account_creation, s.ts) AS account_creation
            FROM src AS s
            INNER JOIN graph_nodes AS g ON s.userid = g.userid
            GROUP BY s.userid
            """
        ).fetch_arrow_table()
    finally:
        conn.unregister("graph_nodes")
    return _arrow_to_raw_by_user(table)


def fetch_profiles_covid19(
    conn: Any,
    scan_sql: str,
    user_ids: Sequence[int],
) -> dict[int, dict[str, Any]]:
    """covid19: profiles from both the author ``user`` struct and the retweeted
    ``retweeted_status.user`` struct, latest per user by tweet ``created_at``.

    Struct fields are accessed with bracket notation so the ``user`` column is
    not mistaken for a table alias.
    """
    _register_graph_nodes(conn, user_ids)
    try:
        table = conn.execute(
            f"""
            WITH prof AS (
                SELECT
                    CAST("user"['id'] AS BIGINT)                       AS userid,
                    TRY_CAST("user"['followers_count']  AS DOUBLE)     AS followers_count,
                    TRY_CAST("user"['friends_count']    AS DOUBLE)     AS friends_count,
                    TRY_CAST("user"['statuses_count']   AS DOUBLE)     AS statuses_count,
                    TRY_CAST("user"['favourites_count'] AS DOUBLE)     AS favourites_count,
                    TRY_CAST("user"['listed_count']     AS DOUBLE)     AS listed_count,
                    CAST("user"['created_at'] AS VARCHAR)              AS account_creation,
                    CAST(created_at AS VARCHAR)                        AS ts
                FROM {scan_sql}
                WHERE "user"['id'] IS NOT NULL
                UNION ALL
                SELECT
                    CAST(retweeted_status['user']['id'] AS BIGINT)                   AS userid,
                    TRY_CAST(retweeted_status['user']['followers_count']  AS DOUBLE) AS followers_count,
                    TRY_CAST(retweeted_status['user']['friends_count']    AS DOUBLE) AS friends_count,
                    TRY_CAST(retweeted_status['user']['statuses_count']   AS DOUBLE) AS statuses_count,
                    TRY_CAST(retweeted_status['user']['favourites_count'] AS DOUBLE) AS favourites_count,
                    TRY_CAST(retweeted_status['user']['listed_count']     AS DOUBLE) AS listed_count,
                    CAST(retweeted_status['user']['created_at'] AS VARCHAR)          AS account_creation,
                    CAST(created_at AS VARCHAR)                                      AS ts
                FROM {scan_sql}
                WHERE retweeted_status['user']['id'] IS NOT NULL
            )
            SELECT
                p.userid,
                arg_max(p.followers_count,  p.ts) AS followers_count,
                arg_max(p.friends_count,    p.ts) AS friends_count,
                arg_max(p.statuses_count,   p.ts) AS statuses_count,
                arg_max(p.favourites_count, p.ts) AS favourites_count,
                arg_max(p.listed_count,     p.ts) AS listed_count,
                arg_max(p.account_creation, p.ts) AS account_creation
            FROM prof AS p
            INNER JOIN graph_nodes AS g ON p.userid = g.userid
            GROUP BY p.userid
            """
        ).fetch_arrow_table()
    finally:
        conn.unregister("graph_nodes")
    return _arrow_to_raw_by_user(table)


def fetch_profiles_twibot20(
    node_json_path: str,
    user_ids: Sequence[str],
    *,
    id_to_raw: Optional[Mapping[str, Any]] = None,
) -> dict[Any, dict[str, Any]]:
    """twibot20: ``node.json`` ``public_metrics`` block.

    TwiBot-20 ids are strings like ``"u17461978"`` throughout the graph
    (``user_ids``, edges, labels), so the result is keyed by that string id to
    match directly. ``id_to_raw`` may remap the node.json id to a different graph
    id if ever needed. ``favourites_count`` has no v2 equivalent and is left
    missing (NaN downstream).
    """
    wanted: Optional[set] = {str(u) for u in user_ids} if user_ids is not None else None
    out: dict[Any, dict[str, Any]] = {}

    with open(node_json_path, "r", encoding="utf-8") as handle:
        nodes = json.load(handle)

    for rec in nodes:
        if not isinstance(rec, dict):
            continue
        str_id = rec.get("id")
        if str_id is None:
            continue
        key = id_to_raw.get(str(str_id)) if id_to_raw is not None else str(str_id)
        if key is None or (wanted is not None and str(key) not in wanted):
            continue
        pm = rec.get("public_metrics") or {}
        entry: dict[str, Any] = {}
        if pm.get("followers_count") is not None:
            entry["followers_count"] = pm["followers_count"]
        if pm.get("following_count") is not None:
            entry["friends_count"] = pm["following_count"]
        if pm.get("tweet_count") is not None:
            entry["statuses_count"] = pm["tweet_count"]
        if pm.get("listed_count") is not None:
            entry["listed_count"] = pm["listed_count"]
        if rec.get("created_at") is not None:
            entry["account_creation"] = rec["created_at"]
        out[key] = entry

    return out


ADAPTERS = {
    "midterm": "flat",
    "ukr_rus_twitter": "flat",
    "covid19_twitter": "covid19",
    "twibot20": "twibot20",
    # cp_hk_twitter intentionally absent: no profile metrics in source.
}


__all__ = [
    "fetch_profiles_flat",
    "fetch_profiles_covid19",
    "fetch_profiles_twibot20",
    "ADAPTERS",
]
