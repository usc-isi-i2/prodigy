from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest

from scripts.bio_embeddings.preprocessing import (
    bio_hash,
    extract_role_observations,
    normalize_bio_text,
)


TWITTER_DATE_0 = "Tue Feb 22 04:00:00 +0000 2022"
TWITTER_DATE_1 = "Tue Feb 22 05:00:00 +0000 2022"
TWITTER_DATE_2 = "Tue Feb 22 06:00:00 +0000 2022"
TWITTER_DATE_3 = "Tue Feb 22 07:00:00 +0000 2022"


def test_normalization_and_hashing() -> None:
    normalized = normalize_bio_text("  Cafe\u0301 @SomeOne\nhttps://example.com/x  ")
    assert normalized == "Café <USER> <URL>"
    assert bio_hash(normalized) == bio_hash("Café <USER> <URL>")


def test_role_extraction_and_retweeted_quote_split() -> None:
    quote_row = {
        "userid": "u1",
        "description": "author",
        "qtd_userid": "q1",
        "qtd_user_description": "quoted",
        "tweet_type": "quote",
    }
    retweeted_quote_row = {
        "userid": "u1",
        "description": "author",
        "rt_tweetid": "rt123",
        "rt_userid": "rt1",
        "rt_user_description": "rt",
        "qtd_userid": "q2",
        "qtd_user_description": "retweeted quoted",
        "tweet_type": "retweet",
    }
    assert [row["source_role"] for row in extract_role_observations(quote_row)] == [
        "author",
        "quoted_author",
    ]
    assert [row["source_role"] for row in extract_role_observations(retweeted_quote_row)] == [
        "author",
        "retweeted_author",
        "retweeted_quoted_author",
    ]


def _parquet_modules():
    pyarrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    return pyarrow, parquet


def _write_fixture(input_root: Path) -> list[dict[str, object]]:
    pa, pq = _parquet_modules()
    partition = input_root / "2022-02"
    partition.mkdir(parents=True)
    path = partition / "fixture.parquet"
    rows = [
        {
            "tweetid": "1",
            "date": TWITTER_DATE_0,
            "tweet_type": "original",
            "text": "hello",
            "userid": "u1",
            "description": "Ukraine news @abc https://example.com",
            "rt_userid": None,
            "rt_user_description": None,
            "rt_tweetid": None,
            "rt_text": None,
            "qtd_userid": None,
            "qtd_user_description": None,
            "qtd_tweetid": None,
            "qtd_text": None,
        },
        {
            "tweetid": "2",
            "date": TWITTER_DATE_1,
            "tweet_type": "quote",
            "text": "quote",
            "userid": "u2",
            "description": "Ukraine news @xyz https://example.org",
            "rt_userid": None,
            "rt_user_description": None,
            "rt_tweetid": None,
            "rt_text": None,
            "qtd_userid": "q1",
            "qtd_user_description": "Quoted analyst",
            "qtd_tweetid": "q100",
            "qtd_text": "quoted text",
        },
        {
            "tweetid": "3",
            "date": TWITTER_DATE_2,
            "tweet_type": "retweet",
            "text": "RT @rt1: something",
            "userid": "u1",
            "description": "Changed bio",
            "rt_userid": "rt1",
            "rt_user_description": "Retweeted author",
            "rt_tweetid": "rt100",
            "rt_text": "retweeted text",
            "qtd_userid": "q2",
            "qtd_user_description": "Nested quote author",
            "qtd_tweetid": "q200",
            "qtd_text": "nested quoted text",
        },
        {
            "tweetid": "4",
            "date": TWITTER_DATE_3,
            "tweet_type": "retweet",
            "text": "RT @rt1: something again",
            "userid": "u3",
            "description": "   ",
            "rt_userid": "rt1",
            "rt_user_description": "Retweeted author",
            "rt_tweetid": "rt101",
            "rt_text": "retweeted text again",
            "qtd_userid": None,
            "qtd_user_description": None,
            "qtd_tweetid": None,
            "qtd_text": None,
        },
        {
            "tweetid": "5",
            "date": "not a date",
            "tweet_type": "quote",
            "text": "quote again",
            "userid": "u1",
            "description": "Changed bio",
            "rt_userid": None,
            "rt_user_description": None,
            "rt_tweetid": None,
            "rt_text": None,
            "qtd_userid": "q1",
            "qtd_user_description": "Quoted analyst",
            "qtd_tweetid": "q101",
            "qtd_text": "quoted text again",
        },
    ]
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path, compression="zstd")
    return [
        {
            "source_file_index": 0,
            "relative_path": "2022-02/fixture.parquet",
            "row_count": len(rows),
            "first_global_row_id": 0,
            "size_bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
            "sha256": "",
        }
    ]


def test_duckdb_bio_index_fixture(tmp_path: Path) -> None:
    _, pq = _parquet_modules()
    pytest.importorskip("duckdb")
    from scripts.bio_embeddings.indexer import build_bio_index

    input_root = tmp_path / "input"
    output_root = tmp_path / "out"
    source_rows = _write_fixture(input_root)
    cfg = {
        "duckdb_temp_dir": str(tmp_path / "duckdb_tmp"),
        "duckdb_memory_limit": "",
        "duckdb_threads": 1,
        "normalization_batch_size": 2,
        "index_parquet_row_group_size": 2,
        "keep_work_dir": False,
    }
    logger = logging.getLogger("test-bio-index")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())

    summary = build_bio_index(input_root, output_root, source_rows, cfg, logger)
    bio_texts = pq.read_table(output_root / "bio_texts.parquet").to_pylist()
    user_bios = pq.read_table(output_root / "user_bio_observations.parquet").to_pylist()

    assert summary["source_rows"] == 5
    assert summary["bio_observations"] == 10
    assert summary["distinct_bio_texts"] == 5
    assert summary["user_bio_pairs"] == 6
    assert summary["invalid_date_observations"] == 2
    assert not (output_root / "_work").exists()

    bio_hashes = [row["bio_hash"] for row in bio_texts]
    assert bio_hashes == sorted(bio_hashes)
    assert len(bio_hashes) == len(set(bio_hashes))
    assert {row["normalized_bio_text"] for row in bio_texts} == {
        "Ukraine news <USER> <URL>",
        "Changed bio",
        "Quoted analyst",
        "Retweeted author",
        "Nested quote author",
    }

    by_user_bio = {(row["userid"], row["bio_hash"]): row for row in user_bios}
    changed_hash = bio_hash(normalize_bio_text("Changed bio"))
    quoted_hash = bio_hash(normalize_bio_text("Quoted analyst"))
    rt_hash = bio_hash(normalize_bio_text("Retweeted author"))
    nested_hash = bio_hash(normalize_bio_text("Nested quote author"))

    assert by_user_bio[("u1", changed_hash)]["n_author_observations"] == 2
    assert by_user_bio[("q1", quoted_hash)]["n_quoted_author_observations"] == 2
    assert by_user_bio[("rt1", rt_hash)]["n_retweeted_author_observations"] == 2
    assert by_user_bio[("q2", nested_hash)]["source_roles"] == "retweeted_quoted_author"


def test_validate_basic_shard(tmp_path: Path) -> None:
    _parquet_modules()
    from scripts.bio_embeddings.schemas import BIO_SHARD_META_SCHEMA
    from scripts.bio_embeddings.worker import shard_paths, validate_basic_shard
    from scripts.tweet_embeddings.io_utils import (
        atomic_write_npy,
        atomic_write_parquet,
        sha256_file,
    )

    output_root = tmp_path / "out"
    (output_root / "shards").mkdir(parents=True)
    paths = shard_paths(output_root, 0)
    embeddings = np.zeros((2, 768), dtype=np.float16)
    embeddings[:, 0] = 1
    emb_sha = atomic_write_npy(paths["emb"], embeddings)
    meta_sha = atomic_write_parquet(
        paths["meta"],
        [
            {
                "bio_id": 0,
                "bio_hash": "a" * 64,
                "embedding_row": 0,
                "token_length": 3,
                "truncation_flag": False,
                "embedding_status": "embedded",
            },
            {
                "bio_id": 1,
                "bio_hash": "b" * 64,
                "embedding_row": 1,
                "token_length": 4,
                "truncation_flag": False,
                "embedding_status": "embedded",
            },
        ],
        BIO_SHARD_META_SCHEMA,
    )
    paths["manifest"].write_text(
        json.dumps(
            {
                "embedding_sha256": emb_sha,
                "metadata_sha256": meta_sha,
                "embedding_dim": 768,
            }
        ),
        encoding="utf-8",
    )

    assert validate_basic_shard(paths, 768) == (True, "ok")
    paths["manifest"].write_text(
        json.dumps(
            {
                "embedding_sha256": "bad",
                "metadata_sha256": sha256_file(paths["meta"]),
                "embedding_dim": 768,
            }
        ),
        encoding="utf-8",
    )
    valid, reason = validate_basic_shard(paths, 768)
    assert not valid
    assert reason == "embedding_checksum_mismatch"
