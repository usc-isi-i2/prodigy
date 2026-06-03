"""PyArrow schemas for bio embedding artifacts."""

import pyarrow as pa


NORMALIZED_RAW_BIO_SCHEMA = pa.schema(
    [
        ("raw_bio_text", pa.string()),
        ("normalized_bio_text", pa.string()),
        ("bio_hash", pa.string()),
        ("is_empty_after_normalization", pa.bool_()),
    ]
)

BIO_TEXT_SCHEMA = pa.schema(
    [
        ("bio_id", pa.int64()),
        ("bio_hash", pa.string()),
        ("normalized_bio_text", pa.string()),
        ("n_observations", pa.int64()),
        ("first_seen_at", pa.timestamp("us")),
        ("last_seen_at", pa.timestamp("us")),
    ]
)

BIO_SHARD_META_SCHEMA = pa.schema(
    [
        ("bio_id", pa.int64()),
        ("bio_hash", pa.string()),
        ("embedding_row", pa.int64()),
        ("token_length", pa.int32()),
        ("truncation_flag", pa.bool_()),
        ("embedding_status", pa.string()),
    ]
)

BIO_EMBEDDING_INDEX_SCHEMA = pa.schema(
    [
        ("bio_id", pa.int64()),
        ("bio_hash", pa.string()),
        ("embedding_shard", pa.string()),
        ("embedding_row", pa.int64()),
        ("embedding_dim", pa.int32()),
        ("embedding_dtype", pa.string()),
        ("model", pa.string()),
        ("revision", pa.string()),
    ]
)
