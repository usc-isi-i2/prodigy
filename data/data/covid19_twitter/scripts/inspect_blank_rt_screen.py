#!/usr/bin/env python3
"""Inspect parquet rows where retweet target handles are missing."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _require_duckdb():
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("duckdb is required for parquet inspection") from exc
    return duckdb


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect covid19_twitter parquet rows with blank retweet target handles."
    )
    parser.add_argument(
        "--parquet-root",
        default="/dataMeR1/phil/data/covid19_twitter/parquet",
    )
    parser.add_argument("--sample", type=int, default=20)
    parser.add_argument("--duckdb-threads", type=int, default=16)
    args = parser.parse_args()

    parquet_glob = str(Path(args.parquet_root) / "*.parquet")
    duckdb = _require_duckdb()
    conn = duckdb.connect(database=":memory:")
    try:
        if args.duckdb_threads > 0:
            conn.execute(f"SET threads={int(args.duckdb_threads)}")

        query = f"""
        WITH source AS (
            SELECT
                COALESCE(
                    CAST(userid AS VARCHAR),
                    json_extract_string(to_json("user"), '$.id_str'),
                    json_extract_string(to_json("user"), '$.id')
                ) AS userid,
                COALESCE(
                    CAST(screen_name AS VARCHAR),
                    json_extract_string(to_json("user"), '$.screen_name')
                ) AS screen_name,
                COALESCE(
                    CAST(rt_userid AS VARCHAR),
                    json_extract_string(to_json(retweeted_status), '$.user.id_str'),
                    json_extract_string(to_json(retweeted_status), '$.user.id')
                ) AS rt_userid,
                COALESCE(
                    CAST(rt_screen AS VARCHAR),
                    json_extract_string(to_json(retweeted_status), '$.user.screen_name')
                ) AS rt_screen,
                COALESCE(
                    CAST(rt_text AS VARCHAR),
                    json_extract_string(to_json(retweeted_status), '$.text')
                ) AS rt_text,
                COALESCE(CAST(tweet_type AS VARCHAR), '') AS tweet_type,
                COALESCE(CAST(text AS VARCHAR), '') AS text,
                COALESCE(CAST(tweetid AS VARCHAR), CAST(id_str AS VARCHAR), CAST(id AS VARCHAR)) AS tweetid
            FROM read_parquet({_sql_literal(parquet_glob)})
        ),
        flagged AS (
            SELECT *
            FROM source
            WHERE (
                lower(tweet_type) LIKE '%retweet%'
                OR NULLIF(trim(COALESCE(rt_userid, '')), '') IS NOT NULL
                OR NULLIF(trim(COALESCE(rt_text, '')), '') IS NOT NULL
                OR text LIKE 'RT %'
            )
            AND NULLIF(trim(COALESCE(rt_screen, '')), '') IS NULL
        )
        SELECT * FROM flagged
        LIMIT {int(args.sample)}
        """

        summary = conn.execute(
            f"""
            WITH source AS (
                SELECT
                    COALESCE(CAST(rt_userid AS VARCHAR), json_extract_string(to_json(retweeted_status), '$.user.id_str')) AS rt_userid,
                    COALESCE(CAST(rt_screen AS VARCHAR), json_extract_string(to_json(retweeted_status), '$.user.screen_name')) AS rt_screen,
                    COALESCE(CAST(rt_text AS VARCHAR), json_extract_string(to_json(retweeted_status), '$.text')) AS rt_text,
                    COALESCE(CAST(tweet_type AS VARCHAR), '') AS tweet_type,
                    COALESCE(CAST(text AS VARCHAR), '') AS text
                FROM read_parquet({_sql_literal(parquet_glob)})
            )
            SELECT
                count(*) FILTER (
                    WHERE lower(tweet_type) LIKE '%retweet%'
                        OR NULLIF(trim(COALESCE(rt_userid, '')), '') IS NOT NULL
                        OR NULLIF(trim(COALESCE(rt_text, '')), '') IS NOT NULL
                        OR text LIKE 'RT %'
                ) AS retweet_like_rows,
                count(*) FILTER (
                    WHERE (
                        lower(tweet_type) LIKE '%retweet%'
                        OR NULLIF(trim(COALESCE(rt_userid, '')), '') IS NOT NULL
                        OR NULLIF(trim(COALESCE(rt_text, '')), '') IS NOT NULL
                        OR text LIKE 'RT %'
                    )
                    AND NULLIF(trim(COALESCE(rt_screen, '')), '') IS NULL
                ) AS blank_rt_screen_rows
            FROM source
            """
        ).fetchone()

        print(f"retweet_like_rows={int(summary[0]):,}")
        print(f"blank_rt_screen_rows={int(summary[1]):,}")
        print()
        rows = conn.execute(query).fetchall()
        for row in rows:
            print(row)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
