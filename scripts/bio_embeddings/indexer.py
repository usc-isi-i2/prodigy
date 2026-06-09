"""DuckDB-backed bio text and user-bio provenance indexing."""

from __future__ import annotations

import logging
from pathlib import Path
import shutil
import socket
import os
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.tweet_embeddings.io_utils import (
    atomic_replace,
    atomic_write_json,
    sha256_file,
    utc_now,
)

from .constants import (
    PREPROCESSING_VERSION,
    ROLE_AUTHOR,
    ROLE_ORDER,
    ROLE_QUOTED_AUTHOR,
    ROLE_RETWEETED_AUTHOR,
    ROLE_RETWEETED_QUOTED_AUTHOR,
)
from .preprocessing import bio_hash, normalize_bio_text
from .schemas import NORMALIZED_RAW_BIO_SCHEMA


def _require_duckdb() -> Any:
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - exercised only without dependency.
        raise RuntimeError(
            "duckdb is required for bio indexing. Install "
            "scripts/bio_embeddings/requirements-embeddings.txt."
        ) from exc
    return duckdb


def _sql_literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _copy_query_to_parquet(conn: Any, query: str, final_path: Path, row_group_size: int) -> str:
    tmp_path = final_path.with_name(
        f"{final_path.name}.tmp.{socket.gethostname()}.{os.getpid()}"
    )
    if tmp_path.exists():
        tmp_path.unlink()
    final_path.parent.mkdir(parents=True, exist_ok=True)
    conn.execute(
        f"""
        COPY (
            {query}
        )
        TO {_sql_literal(tmp_path)}
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE {int(row_group_size)})
        """
    )
    atomic_replace(tmp_path, final_path)
    return sha256_file(final_path)


def _write_source_files_table(conn: Any, input_root: Path, source_rows: list[dict[str, Any]]) -> None:
    rows = []
    for row in source_rows:
        rel = str(row["relative_path"])
        rows.append(
            {
                **row,
                "absolute_path": (input_root / rel).as_posix(),
            }
        )
    table = pa.Table.from_pylist(rows)
    conn.register("source_files_arrow", table)
    conn.execute("CREATE OR REPLACE TABLE source_files AS SELECT * FROM source_files_arrow")
    conn.unregister("source_files_arrow")


def _configure_duckdb(conn: Any, cfg: dict[str, Any], work_dir: Path) -> None:
    temp_dir = Path(str(cfg.get("duckdb_temp_dir") or work_dir / "duckdb_tmp"))
    temp_dir.mkdir(parents=True, exist_ok=True)
    conn.execute(f"SET temp_directory={_sql_literal(temp_dir)}")
    if cfg.get("duckdb_memory_limit"):
        conn.execute(f"SET memory_limit={_sql_literal(str(cfg['duckdb_memory_limit']))}")
    if int(cfg.get("duckdb_threads") or 0) > 0:
        conn.execute(f"SET threads={int(cfg['duckdb_threads'])}")


def _source_columns(conn: Any, input_root: Path) -> set[str]:
    parquet_glob = input_root.as_posix().rstrip("/") + "/**/*.parquet"
    rows = conn.execute(
        f"DESCRIBE SELECT * FROM read_parquet({_sql_literal(parquet_glob)})"
    ).fetchall()
    return {str(row[0]) for row in rows}


def _column_exists(columns: set[str], expr: str) -> bool:
    return expr.split(".", 1)[0] in columns


def _coalesce_varchar(columns: set[str], candidates: list[str]) -> str:
    exprs = [f"CAST({expr} AS VARCHAR)" for expr in candidates if _column_exists(columns, expr)]
    if not exprs:
        return "NULL"
    if len(exprs) == 1:
        return exprs[0]
    return f"COALESCE({', '.join(exprs)})"


def _first_varchar(columns: set[str], candidates: list[str]) -> str:
    for expr in candidates:
        if _column_exists(columns, expr):
            return f"CAST({expr} AS VARCHAR)"
    return "NULL"


def _nonempty_varchar_predicate(columns: set[str], candidates: list[str]) -> str:
    predicates = [
        f"NULLIF(trim(COALESCE(CAST({expr} AS VARCHAR), '')), '') IS NOT NULL"
        for expr in candidates
        if _column_exists(columns, expr)
    ]
    if not predicates:
        return "FALSE"
    return "(" + " OR ".join(predicates) + ")"


def _timestamp_expr(columns: set[str]) -> str:
    exprs: list[str] = []
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


def _create_source_scan(conn: Any, input_root: Path) -> None:
    parquet_glob = input_root.as_posix().rstrip("/") + "/**/*.parquet"
    columns = _source_columns(conn, input_root)
    tweetid_expr = _coalesce_varchar(columns, ["tweetid", "id_str", "id", "tweet_id", "tweet_id_num"])
    date_expr = _first_varchar(columns, ["created_at", "date", "created_ts"])
    tweet_text_expr = _coalesce_varchar(columns, ["extended_tweet.full_text", "full_text", "text"])
    userid_expr = _coalesce_varchar(columns, ["userid", "user.id_str", "user.id", "user_id"])
    description_expr = _coalesce_varchar(columns, ["description", "user.description", "user_description"])
    rt_userid_expr = _coalesce_varchar(
        columns,
        ["rt_userid", "retweeted_status.user.id_str", "retweeted_status.user.id", "retweeted_user_id"],
    )
    rt_user_description_expr = _coalesce_varchar(
        columns,
        ["rt_user_description", "retweeted_status.user.description"],
    )
    rt_tweetid_expr = _coalesce_varchar(
        columns,
        ["rt_tweetid", "retweeted_status.id_str", "retweeted_status.id", "retweeted_tweet_id"],
    )
    rt_text_expr = _coalesce_varchar(
        columns,
        [
            "rt_text",
            "retweeted_status.extended_tweet.full_text",
            "retweeted_status.full_text",
            "retweeted_status.text",
            "retweeted_text",
        ],
    )
    qtd_userid_expr = _coalesce_varchar(
        columns,
        [
            "qtd_userid",
            "retweeted_status.quoted_status.user.id_str",
            "retweeted_status.quoted_status.user.id",
            "quoted_status.user.id_str",
            "quoted_status.user.id",
        ],
    )
    qtd_user_description_expr = _coalesce_varchar(
        columns,
        [
            "qtd_user_description",
            "retweeted_status.quoted_status.user.description",
            "quoted_status.user.description",
        ],
    )
    qtd_tweetid_expr = _coalesce_varchar(
        columns,
        [
            "qtd_tweetid",
            "retweeted_status.quoted_status.id_str",
            "retweeted_status.quoted_status.id",
            "quoted_status.id_str",
            "quoted_status.id",
        ],
    )
    qtd_text_expr = _coalesce_varchar(
        columns,
        [
            "qtd_text",
            "retweeted_status.quoted_status.extended_tweet.full_text",
            "retweeted_status.quoted_status.full_text",
            "retweeted_status.quoted_status.text",
            "quoted_status.extended_tweet.full_text",
            "quoted_status.full_text",
            "quoted_status.text",
        ],
    )
    observed_at_expr = _timestamp_expr(columns)
    retweet_presence_predicate = _nonempty_varchar_predicate(
        columns,
        ["rt_tweetid", "retweeted_status.id_str", "retweeted_status.id", "retweeted_tweet_id"],
    )
    quote_presence_predicate = (
        _nonempty_varchar_predicate(
            columns,
            [
                "qtd_tweetid",
                "retweeted_status.quoted_status.id_str",
                "retweeted_status.quoted_status.id",
                "quoted_status.id_str",
                "quoted_status.id",
            ],
        )
    )
    if "is_quote_status" in columns:
        quote_presence_predicate = f"({quote_presence_predicate} OR COALESCE(try_cast(is_quote_status AS BOOLEAN), FALSE))"
    derived_tweet_type_expr = (
        "CASE "
        f"WHEN {retweet_presence_predicate} OR COALESCE({tweet_text_expr}, '') LIKE 'RT %' THEN 'retweet' "
        f"WHEN {quote_presence_predicate} THEN 'quote' "
        "ELSE 'original' END"
    )
    tweet_type_expr = (
        f"COALESCE(CAST(tweet_type AS VARCHAR), {derived_tweet_type_expr})"
        if "tweet_type" in columns
        else derived_tweet_type_expr
    )
    conn.execute(
        f"""
        CREATE OR REPLACE VIEW source_scan AS
        SELECT
            CAST(s.source_file_index AS INTEGER) AS source_file_index,
            CAST(s.relative_path AS VARCHAR) AS source_file,
            CAST(p.file_row_number AS BIGINT) AS source_offset,
            CAST(s.first_global_row_id + p.file_row_number AS BIGINT) AS global_row_id,
            {tweetid_expr} AS tweetid,
            {date_expr} AS date,
            {observed_at_expr} AS observed_at,
            {userid_expr} AS userid,
            {description_expr} AS description,
            {rt_userid_expr} AS rt_userid,
            {rt_user_description_expr} AS rt_user_description,
            {rt_tweetid_expr} AS rt_tweetid,
            {rt_text_expr} AS rt_text,
            {qtd_userid_expr} AS qtd_userid,
            {qtd_user_description_expr} AS qtd_user_description,
            {qtd_tweetid_expr} AS qtd_tweetid,
            {qtd_text_expr} AS qtd_text,
            {tweet_type_expr} AS tweet_type,
            {tweet_text_expr} AS text,
            (
                lower(COALESCE({tweet_type_expr}, '')) LIKE '%retweet%'
                OR NULLIF(trim(COALESCE({rt_tweetid_expr}, '')), '') IS NOT NULL
                OR NULLIF(trim(COALESCE({rt_text_expr}, '')), '') IS NOT NULL
                OR COALESCE({tweet_text_expr}, '') LIKE 'RT %'
            ) AS is_retweet_like,
            (
                NULLIF(trim(COALESCE({qtd_userid_expr}, '')), '') IS NOT NULL
                OR NULLIF(trim(COALESCE({qtd_user_description_expr}, '')), '') IS NOT NULL
                OR NULLIF(trim(COALESCE({qtd_tweetid_expr}, '')), '') IS NOT NULL
                OR NULLIF(trim(COALESCE({qtd_text_expr}, '')), '') IS NOT NULL
            ) AS has_quoted_author
        FROM read_parquet(
            {_sql_literal(parquet_glob)},
            filename=true,
            file_row_number=true
        ) AS p
        INNER JOIN source_files AS s
            ON p.filename = s.absolute_path
        """
    )


def _create_raw_observations(conn: Any) -> None:
    conn.execute(
        f"""
        CREATE OR REPLACE TABLE raw_bio_observations AS
        SELECT
            { _sql_literal(ROLE_AUTHOR) } AS source_role,
            userid,
            description AS raw_bio_text,
            tweetid,
            date,
            observed_at,
            source_file,
            source_file_index,
            source_offset,
            global_row_id
        FROM source_scan

        UNION ALL

        SELECT
            { _sql_literal(ROLE_RETWEETED_AUTHOR) } AS source_role,
            rt_userid AS userid,
            rt_user_description AS raw_bio_text,
            tweetid,
            date,
            observed_at,
            source_file,
            source_file_index,
            source_offset,
            global_row_id
        FROM source_scan
        WHERE is_retweet_like
            OR NULLIF(trim(COALESCE(rt_userid, '')), '') IS NOT NULL
            OR NULLIF(trim(COALESCE(rt_user_description, '')), '') IS NOT NULL

        UNION ALL

        SELECT
            CASE
                WHEN is_retweet_like THEN { _sql_literal(ROLE_RETWEETED_QUOTED_AUTHOR) }
                ELSE { _sql_literal(ROLE_QUOTED_AUTHOR) }
            END AS source_role,
            qtd_userid AS userid,
            qtd_user_description AS raw_bio_text,
            tweetid,
            date,
            observed_at,
            source_file,
            source_file_index,
            source_offset,
            global_row_id
        FROM source_scan
        WHERE has_quoted_author
        """
    )


def _iter_duckdb_batches(result: Any, batch_size: int) -> Iterable[pa.RecordBatch]:
    if hasattr(result, "fetch_record_batch"):
        reader = result.fetch_record_batch(rows_per_batch=batch_size)
        for batch in reader:
            yield batch
        return

    while True:  # pragma: no cover - compatibility fallback for older DuckDB.
        frame = result.fetch_df_chunk()
        if frame is None or frame.empty:
            break
        yield pa.RecordBatch.from_pandas(frame, preserve_index=False)


def _write_normalized_raw_bios(
    conn: Any,
    work_dir: Path,
    batch_size: int,
    logger: logging.Logger,
) -> Path:
    normalized_path = work_dir / "normalized_raw_bios.parquet"
    tmp_path = normalized_path.with_name(
        f"{normalized_path.name}.tmp.{socket.gethostname()}.{os.getpid()}"
    )
    if tmp_path.exists():
        tmp_path.unlink()

    conn.execute(
        """
        CREATE OR REPLACE TABLE distinct_raw_bios AS
        SELECT DISTINCT CAST(raw_bio_text AS VARCHAR) AS raw_bio_text
        FROM raw_bio_observations
        WHERE raw_bio_text IS NOT NULL
            AND trim(CAST(raw_bio_text AS VARCHAR)) <> ''
        """
    )
    total = int(conn.execute("SELECT count(*) FROM distinct_raw_bios").fetchone()[0])
    logger.info("normalizing distinct raw bio texts=%s", total)

    writer = pq.ParquetWriter(tmp_path, NORMALIZED_RAW_BIO_SCHEMA, compression="zstd")
    written = 0
    try:
        result = conn.execute("SELECT raw_bio_text FROM distinct_raw_bios ORDER BY raw_bio_text")
        for batch in _iter_duckdb_batches(result, batch_size):
            raw_values = batch.column(0).to_pylist()
            normalized_values = [normalize_bio_text(value) for value in raw_values]
            hashes = [bio_hash(value) if value else "" for value in normalized_values]
            empty_flags = [not bool(value) for value in normalized_values]
            table = pa.Table.from_pydict(
                {
                    "raw_bio_text": raw_values,
                    "normalized_bio_text": normalized_values,
                    "bio_hash": hashes,
                    "is_empty_after_normalization": empty_flags,
                },
                schema=NORMALIZED_RAW_BIO_SCHEMA,
            )
            writer.write_table(table)
            written += len(raw_values)
            if written % max(batch_size * 10, 1) == 0:
                logger.info("normalized raw bio texts=%s/%s", written, total)
    finally:
        writer.close()

    atomic_replace(tmp_path, normalized_path)
    logger.info("normalized raw bio text map written rows=%s path=%s", written, normalized_path)
    return normalized_path


def _create_valid_observations(conn: Any, normalized_path: Path) -> None:
    conn.execute(
        f"""
        CREATE OR REPLACE TABLE normalized_raw_bios AS
        SELECT *
        FROM read_parquet({_sql_literal(normalized_path)})
        """
    )
    conn.execute(
        """
        CREATE OR REPLACE TABLE valid_bio_observations AS
        SELECT
            r.source_role,
            r.userid,
            n.bio_hash,
            n.normalized_bio_text,
            r.tweetid,
            r.date,
            r.observed_at,
            r.source_file,
            r.source_file_index,
            r.source_offset,
            r.global_row_id
        FROM raw_bio_observations AS r
        INNER JOIN normalized_raw_bios AS n
            ON CAST(r.raw_bio_text AS VARCHAR) = n.raw_bio_text
        WHERE NOT n.is_empty_after_normalization
        """
    )


def _copy_bio_texts(conn: Any, output_root: Path, row_group_size: int) -> str:
    return _copy_query_to_parquet(
        conn,
        """
        WITH grouped AS (
            SELECT
                bio_hash,
                min(normalized_bio_text) AS normalized_bio_text,
                count(*) AS n_observations,
                min(observed_at) AS first_seen_at,
                max(observed_at) AS last_seen_at
            FROM valid_bio_observations
            GROUP BY bio_hash
        )
        SELECT
            CAST(row_number() OVER (ORDER BY bio_hash) - 1 AS BIGINT) AS bio_id,
            bio_hash,
            normalized_bio_text,
            CAST(n_observations AS BIGINT) AS n_observations,
            first_seen_at,
            last_seen_at
        FROM grouped
        ORDER BY bio_hash
        """,
        output_root / "bio_texts.parquet",
        row_group_size,
    )


def _copy_user_bio_observations(conn: Any, output_root: Path, row_group_size: int) -> str:
    role_expr = """
        regexp_replace(concat(
            CASE WHEN n_author_observations > 0 THEN 'author,' ELSE '' END,
            CASE WHEN n_retweeted_author_observations > 0 THEN 'retweeted_author,' ELSE '' END,
            CASE WHEN n_quoted_author_observations > 0 THEN 'quoted_author,' ELSE '' END,
            CASE WHEN n_retweeted_quoted_author_observations > 0 THEN 'retweeted_quoted_author,' ELSE '' END
        ), ',$', '')
    """
    return _copy_query_to_parquet(
        conn,
        f"""
        WITH filtered AS (
            SELECT *
            FROM valid_bio_observations
            WHERE NULLIF(trim(COALESCE(userid, '')), '') IS NOT NULL
                AND lower(trim(userid)) NOT IN ('nan', 'none', '<na>')
        ),
        ranked AS (
            SELECT
                *,
                row_number() OVER (
                    PARTITION BY userid, bio_hash
                    ORDER BY observed_at ASC NULLS LAST, global_row_id ASC
                ) AS rn_first,
                row_number() OVER (
                    PARTITION BY userid, bio_hash
                    ORDER BY observed_at DESC NULLS LAST, global_row_id DESC
                ) AS rn_last
            FROM filtered
        ),
        aggregated AS (
            SELECT
                userid,
                bio_hash,
                min(observed_at) AS first_seen_at,
                max(observed_at) AS last_seen_at,
                count(*) AS n_observations,
                sum(CASE WHEN source_role = 'author' THEN 1 ELSE 0 END) AS n_author_observations,
                sum(CASE WHEN source_role = 'retweeted_author' THEN 1 ELSE 0 END) AS n_retweeted_author_observations,
                sum(CASE WHEN source_role = 'quoted_author' THEN 1 ELSE 0 END) AS n_quoted_author_observations,
                sum(CASE WHEN source_role = 'retweeted_quoted_author' THEN 1 ELSE 0 END) AS n_retweeted_quoted_author_observations,
                max(CASE WHEN rn_first = 1 THEN tweetid ELSE NULL END) AS first_tweetid,
                max(CASE WHEN rn_last = 1 THEN tweetid ELSE NULL END) AS last_tweetid,
                max(CASE WHEN rn_first = 1 THEN global_row_id ELSE NULL END) AS first_global_row_id,
                max(CASE WHEN rn_last = 1 THEN global_row_id ELSE NULL END) AS last_global_row_id,
                max(CASE WHEN rn_first = 1 THEN source_file ELSE NULL END) AS first_source_file,
                max(CASE WHEN rn_last = 1 THEN source_file ELSE NULL END) AS last_source_file,
                max(CASE WHEN rn_first = 1 THEN source_file_index ELSE NULL END) AS first_source_file_index,
                max(CASE WHEN rn_last = 1 THEN source_file_index ELSE NULL END) AS last_source_file_index,
                max(CASE WHEN rn_first = 1 THEN source_offset ELSE NULL END) AS first_source_offset,
                max(CASE WHEN rn_last = 1 THEN source_offset ELSE NULL END) AS last_source_offset
            FROM ranked
            GROUP BY userid, bio_hash
        )
        SELECT
            userid,
            bio_hash,
            first_seen_at,
            last_seen_at,
            CAST(n_observations AS BIGINT) AS n_observations,
            CAST(n_author_observations AS BIGINT) AS n_author_observations,
            CAST(n_retweeted_author_observations AS BIGINT) AS n_retweeted_author_observations,
            CAST(n_quoted_author_observations AS BIGINT) AS n_quoted_author_observations,
            CAST(n_retweeted_quoted_author_observations AS BIGINT) AS n_retweeted_quoted_author_observations,
            {role_expr} AS source_roles,
            first_tweetid,
            last_tweetid,
            CAST(first_global_row_id AS BIGINT) AS first_global_row_id,
            CAST(last_global_row_id AS BIGINT) AS last_global_row_id,
            first_source_file,
            last_source_file,
            CAST(first_source_file_index AS INTEGER) AS first_source_file_index,
            CAST(last_source_file_index AS INTEGER) AS last_source_file_index,
            CAST(first_source_offset AS BIGINT) AS first_source_offset,
            CAST(last_source_offset AS BIGINT) AS last_source_offset
        FROM aggregated
        ORDER BY userid, bio_hash
        """,
        output_root / "user_bio_observations.parquet",
        row_group_size,
    )


def _fetch_dicts(conn: Any, query: str) -> list[dict[str, Any]]:
    table = conn.execute(query).fetch_arrow_table()
    return table.to_pylist()


ROLE_STAT_COUNT_KEYS = (
    "n_observations",
    "n_null_bio",
    "n_empty_raw_bio",
    "n_nonempty_normalized_bio",
    "n_distinct_bio_hashes",
    "n_user_bio_pairs",
)


def _build_summary(
    conn: Any,
    output_root: Path,
    source_rows: list[dict[str, Any]],
    bio_texts_sha256: str,
    user_bio_sha256: str,
    started_at: str,
) -> dict[str, Any]:
    role_stats = _fetch_dicts(
        conn,
        """
        SELECT
            r.source_role,
            count(*) AS n_observations,
            sum(CASE WHEN r.raw_bio_text IS NULL THEN 1 ELSE 0 END) AS n_null_bio,
            sum(CASE
                WHEN r.raw_bio_text IS NOT NULL
                    AND trim(CAST(r.raw_bio_text AS VARCHAR)) = ''
                THEN 1 ELSE 0
            END) AS n_empty_raw_bio,
            0 AS n_nonempty_normalized_bio,
            0 AS n_distinct_bio_hashes,
            0 AS n_user_bio_pairs
        FROM raw_bio_observations AS r
        GROUP BY r.source_role
        ORDER BY CASE r.source_role
            WHEN 'author' THEN 0
            WHEN 'retweeted_author' THEN 1
            WHEN 'quoted_author' THEN 2
            WHEN 'retweeted_quoted_author' THEN 3
            ELSE 4
        END
        """,
    )
    valid_role_stats = {
        row["source_role"]: row
        for row in _fetch_dicts(
            conn,
            """
            SELECT
                source_role,
                count(*) AS n_nonempty_normalized_bio,
                count(DISTINCT bio_hash) AS n_distinct_bio_hashes,
                count(DISTINCT CASE
                    WHEN NULLIF(trim(COALESCE(userid, '')), '') IS NOT NULL
                        AND lower(trim(userid)) NOT IN ('nan', 'none', '<na>')
                    THEN userid || '|' || bio_hash
                    ELSE NULL
                END) AS n_user_bio_pairs
            FROM valid_bio_observations
            GROUP BY source_role
            """,
        )
    }
    for row in role_stats:
        valid = valid_role_stats.get(row["source_role"], {})
        for key in ("n_nonempty_normalized_bio", "n_distinct_bio_hashes", "n_user_bio_pairs"):
            row[key] = int(valid.get(key) or 0)
        for key in ROLE_STAT_COUNT_KEYS:
            row[key] = int(row.get(key) or 0)

    scalar = conn.execute(
        """
        SELECT
            (SELECT count(*) FROM raw_bio_observations) AS bio_observations,
            (SELECT count(*) FROM distinct_raw_bios) AS distinct_raw_bio_texts,
            (SELECT count(*) FROM normalized_raw_bios WHERE is_empty_after_normalization) AS empty_after_normalization,
            (SELECT count(*) FROM bio_texts) AS distinct_bio_texts,
            (SELECT count(*) FROM user_bio_observations) AS user_bio_pairs,
            (SELECT count(DISTINCT userid) FROM user_bio_observations) AS users_with_bios,
            (SELECT count(*) FROM valid_bio_observations WHERE date IS NOT NULL AND trim(date) <> '' AND observed_at IS NULL) AS invalid_date_observations,
            (SELECT count(*) FROM valid_bio_observations
                WHERE NULLIF(trim(COALESCE(userid, '')), '') IS NULL
                    OR lower(trim(userid)) IN ('nan', 'none', '<na>')
            ) AS missing_user_nonempty_bio_observations
        """
    ).fetchone()
    summary = {
        "created_at": utc_now(),
        "started_at": started_at,
        "output_root": str(output_root),
        "preprocessing_version": PREPROCESSING_VERSION,
        "source_files": len(source_rows),
        "source_rows": sum(int(row["row_count"]) for row in source_rows),
        "bio_observations": int(scalar[0]),
        "distinct_raw_bio_texts": int(scalar[1]),
        "empty_after_normalization": int(scalar[2]),
        "distinct_bio_texts": int(scalar[3]),
        "user_bio_pairs": int(scalar[4]),
        "users_with_bios": int(scalar[5]),
        "invalid_date_observations": int(scalar[6]),
        "missing_user_nonempty_bio_observations": int(scalar[7]),
        "role_order": ROLE_ORDER,
        "role_stats": role_stats,
        "artifacts": {
            "bio_texts.parquet": {
                "path": "bio_texts.parquet",
                "sha256": bio_texts_sha256,
            },
            "user_bio_observations.parquet": {
                "path": "user_bio_observations.parquet",
                "sha256": user_bio_sha256,
            },
        },
    }
    return summary


def build_bio_index(
    input_root: Path,
    output_root: Path,
    source_rows: list[dict[str, Any]],
    cfg: dict[str, Any],
    logger: logging.Logger,
) -> dict[str, Any]:
    """Build bio_texts and user_bio_observations artifacts."""
    duckdb = _require_duckdb()
    output_root.mkdir(parents=True, exist_ok=True)
    work_dir = output_root / "_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    db_path = work_dir / "bio_index.duckdb"
    started_at = utc_now()

    logger.info("building bio index work_dir=%s", work_dir)
    conn = duckdb.connect(str(db_path))
    try:
        _configure_duckdb(conn, cfg, work_dir)
        _write_source_files_table(conn, input_root, source_rows)
        _create_source_scan(conn, input_root)
        _create_raw_observations(conn)
        normalized_path = _write_normalized_raw_bios(
            conn,
            work_dir,
            int(cfg.get("normalization_batch_size") or 250_000),
            logger,
        )
        _create_valid_observations(conn, normalized_path)
        row_group_size = int(cfg.get("index_parquet_row_group_size") or 1_000_000)
        bio_texts_sha = _copy_bio_texts(conn, output_root, row_group_size)
        conn.execute(
            f"""
            CREATE OR REPLACE TABLE bio_texts AS
            SELECT *
            FROM read_parquet({_sql_literal(output_root / "bio_texts.parquet")})
            """
        )
        user_bio_sha = _copy_user_bio_observations(conn, output_root, row_group_size)
        conn.execute(
            f"""
            CREATE OR REPLACE TABLE user_bio_observations AS
            SELECT *
            FROM read_parquet({_sql_literal(output_root / "user_bio_observations.parquet")})
            """
        )
        summary = _build_summary(
            conn,
            output_root,
            source_rows,
            bio_texts_sha,
            user_bio_sha,
            started_at,
        )
        atomic_write_json(output_root / "bio_index_summary.json", summary)
        logger.info(
            "bio index complete distinct_bios=%s user_bio_pairs=%s",
            summary["distinct_bio_texts"],
            summary["user_bio_pairs"],
        )
    finally:
        conn.close()

    if not bool(cfg.get("keep_work_dir")):
        shutil.rmtree(work_dir, ignore_errors=True)
        logger.info("removed bio index work dir=%s", work_dir)
    return summary
