"""Source Parquet discovery and deterministic row indexing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .io_utils import atomic_write_parquet, sha256_file
from .schemas import SOURCE_SCHEMA


def discover_parquet_files(input_root: Path) -> list[Path]:
    return sorted(
        [path for path in input_root.rglob("*.parquet") if path.is_file()],
        key=lambda path: path.relative_to(input_root).as_posix(),
    )


def build_source_index(input_root: Path, checksum: bool) -> list[dict[str, Any]]:
    files = discover_parquet_files(input_root)
    if not files:
        raise FileNotFoundError(f"No parquet files found under {input_root}")

    rows: list[dict[str, Any]] = []
    first_global_row_id = 0
    for index, path in enumerate(files):
        rel = path.relative_to(input_root).as_posix()
        metadata = pq.ParquetFile(path).metadata
        stat = path.stat()
        digest = sha256_file(path) if checksum else ""
        row = {
            "source_file_index": index,
            "relative_path": rel,
            "row_count": int(metadata.num_rows),
            "first_global_row_id": int(first_global_row_id),
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "sha256": digest,
        }
        rows.append(row)
        first_global_row_id += int(metadata.num_rows)
    return rows


def load_source_index(path: Path) -> list[dict[str, Any]]:
    table = pq.read_table(path)
    return table.to_pylist()


def write_source_index(path: Path, rows: list[dict[str, Any]]) -> str:
    return atomic_write_parquet(path, rows, SOURCE_SCHEMA)


def total_source_rows(source_rows: list[dict[str, Any]]) -> int:
    return sum(int(row["row_count"]) for row in source_rows)
