"""PyArrow schemas for pipeline artifacts."""

import pyarrow as pa

META_SCHEMA = pa.schema(
    [
        ("global_row_id", pa.int64()),
        ("tweetid", pa.string()),
        ("userid", pa.string()),
        ("lang", pa.string()),
        ("date", pa.string()),
        ("tweet_type", pa.string()),
        ("source_file", pa.string()),
        ("source_file_index", pa.int32()),
        ("source_offset", pa.int64()),
        ("text_hash", pa.string()),
        ("token_length", pa.int32()),
        ("truncation_flag", pa.bool_()),
        ("embedding_status", pa.string()),
        ("text_source", pa.string()),
        ("rt_tweetid", pa.string()),
        ("qtd_tweetid", pa.string()),
        ("reply_statusid", pa.string()),
    ]
)

SKIP_SCHEMA = pa.schema(
    [
        ("global_row_id", pa.int64()),
        ("tweetid", pa.string()),
        ("userid", pa.string()),
        ("source_file", pa.string()),
        ("source_file_index", pa.int32()),
        ("source_offset", pa.int64()),
        ("skip_reason", pa.string()),
        ("text_hash", pa.string()),
        ("text_source", pa.string()),
        ("tweet_type", pa.string()),
    ]
)

SOURCE_SCHEMA = pa.schema(
    [
        ("source_file_index", pa.int32()),
        ("relative_path", pa.string()),
        ("row_count", pa.int64()),
        ("first_global_row_id", pa.int64()),
        ("size_bytes", pa.int64()),
        ("mtime_ns", pa.int64()),
        ("sha256", pa.string()),
    ]
)
