"""Read deterministic shard slices and assemble text/metadata rows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import pyarrow.parquet as pq

from .constants import TEXT_COLUMNS
from .preprocessing import (
    assemble_text,
    coerce_text,
    preprocess_text,
    skip_reason_for_text,
    text_hash,
)


def existing_columns(path: Path) -> list[str]:
    return pq.ParquetFile(path).schema_arrow.names


def iter_source_slices(
    input_root: Path,
    source_rows: list[dict[str, Any]],
    shard_id: int,
    shard_size: int,
    parquet_batch_rows: int,
) -> Iterable[tuple[dict[str, Any], int, pd.DataFrame]]:
    shard_start = shard_id * shard_size
    shard_end = shard_start + shard_size
    for source in source_rows:
        file_start = int(source["first_global_row_id"])
        file_rows = int(source["row_count"])
        file_end = file_start + file_rows
        if file_end <= shard_start:
            continue
        if file_start >= shard_end:
            break
        local_start = max(0, shard_start - file_start)
        local_end = min(file_rows, shard_end - file_start)
        if local_start >= local_end:
            continue

        path = input_root / str(source["relative_path"])
        columns = [col for col in TEXT_COLUMNS if col in existing_columns(path)]
        parquet_file = pq.ParquetFile(path)
        batch_base = 0
        for batch in parquet_file.iter_batches(batch_size=parquet_batch_rows, columns=columns):
            batch_end = batch_base + batch.num_rows
            if batch_end <= local_start:
                batch_base = batch_end
                continue
            if batch_base >= local_end:
                break
            take_start = max(0, local_start - batch_base)
            take_end = min(batch.num_rows, local_end - batch_base)
            if take_start >= take_end:
                batch_base = batch_end
                continue
            sliced = batch.slice(take_start, take_end - take_start)
            df = sliced.to_pandas()
            for column in TEXT_COLUMNS:
                if column not in df.columns:
                    df[column] = ""
            yield source, batch_base + take_start, df[TEXT_COLUMNS]
            batch_base = batch_end


def prepare_shard_texts(
    input_root: Path,
    source_rows: list[dict[str, Any]],
    shard_id: int,
    cfg: dict[str, Any],
) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    texts: list[str] = []
    metadata_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    shard_size = int(cfg["shard_size"])
    batch_rows = int(cfg["parquet_batch_rows"])

    for source, batch_source_offset, df in iter_source_slices(
        input_root, source_rows, shard_id, shard_size, batch_rows
    ):
        source_index = int(source["source_file_index"])
        first_global = int(source["first_global_row_id"])
        source_file = str(source["relative_path"])
        for local_index, row in enumerate(df.to_dict("records")):
            source_offset = int(batch_source_offset + local_index)
            global_row_id = int(first_global + source_offset)
            raw_text, text_source = assemble_text(row)
            processed_text = preprocess_text(raw_text)
            digest = text_hash(processed_text)
            skip_reason = skip_reason_for_text(processed_text)
            if skip_reason is not None:
                skipped_rows.append(
                    {
                        "global_row_id": global_row_id,
                        "tweetid": coerce_text(row.get("tweetid")),
                        "userid": coerce_text(row.get("userid")),
                        "source_file": source_file,
                        "source_file_index": source_index,
                        "source_offset": source_offset,
                        "skip_reason": skip_reason,
                        "text_hash": digest,
                        "text_source": text_source,
                        "tweet_type": coerce_text(row.get("tweet_type")),
                    }
                )
                continue

            metadata_rows.append(
                {
                    "global_row_id": global_row_id,
                    "tweetid": coerce_text(row.get("tweetid")),
                    "userid": coerce_text(row.get("userid")),
                    "lang": coerce_text(row.get("lang")),
                    "date": coerce_text(row.get("date")),
                    "tweet_type": coerce_text(row.get("tweet_type")),
                    "source_file": source_file,
                    "source_file_index": source_index,
                    "source_offset": source_offset,
                    "text_hash": digest,
                    "token_length": -1,
                    "truncation_flag": False,
                    "embedding_status": "embedded",
                    "text_source": text_source,
                    "rt_tweetid": coerce_text(row.get("rt_tweetid")),
                    "qtd_tweetid": coerce_text(row.get("qtd_tweetid")),
                    "reply_statusid": coerce_text(row.get("reply_statusid")),
                }
            )
            texts.append(processed_text)
    return texts, metadata_rows, skipped_rows
