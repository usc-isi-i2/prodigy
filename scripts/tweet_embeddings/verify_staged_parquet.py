#!/usr/bin/env python3
"""Verify the locally staged Ukraine-Russia tweet Parquet mirror on tucker."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_INPUT_ROOT = "/dataMeR1/phil/data/ukr_rus_twitter/parquet"
EXPECTED_FILES = {"2022-02": 158, "2022-03": 719, "2022-04": 621}
EXPECTED_ROWS = {
    "2022-02": 27_830_486,
    "2022-03": 121_461_576,
    "2022-04": 71_617_254,
}

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


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_files(input_root: Path) -> list[Path]:
    return sorted(
        [path for path in input_root.rglob("*.parquet") if path.is_file()],
        key=lambda path: path.relative_to(input_root).as_posix(),
    )


def partition_for(path: Path, input_root: Path) -> str:
    rel = path.relative_to(input_root)
    return rel.parts[0] if rel.parts else ""


def build_source_rows(input_root: Path, files: list[Path], checksum: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    first_global_row_id = 0
    for index, path in enumerate(files):
        rel = path.relative_to(input_root).as_posix()
        metadata = pq.ParquetFile(path).metadata
        stat = path.stat()
        digest = sha256_file(path) if checksum else ""
        rows.append(
            {
                "source_file_index": index,
                "relative_path": rel,
                "row_count": int(metadata.num_rows),
                "first_global_row_id": int(first_global_row_id),
                "size_bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "sha256": digest,
            }
        )
        first_global_row_id += int(metadata.num_rows)
    return rows


def write_source_files(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=SOURCE_SCHEMA)
    pq.write_table(table, path, compression="zstd")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify staged Parquet counts and optionally write source_files.parquet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-root", default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--checksum", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-source-files", default="")
    parser.add_argument("--summary-json", default="")
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    input_root = Path(args.input_root)
    files = discover_files(input_root)
    if not files:
        raise FileNotFoundError(f"No parquet files found under {input_root}")

    source_rows = build_source_rows(input_root, files, checksum=bool(args.checksum))
    file_counts = {partition: 0 for partition in EXPECTED_FILES}
    row_counts = {partition: 0 for partition in EXPECTED_ROWS}
    for path, row in zip(files, source_rows):
        partition = partition_for(path, input_root)
        file_counts[partition] = file_counts.get(partition, 0) + 1
        row_counts[partition] = row_counts.get(partition, 0) + int(row["row_count"])

    errors: list[str] = []
    for partition, expected in EXPECTED_FILES.items():
        actual = file_counts.get(partition, 0)
        if actual != expected:
            errors.append(f"{partition} file_count expected {expected} got {actual}")
    for partition, expected in EXPECTED_ROWS.items():
        actual = row_counts.get(partition, 0)
        if actual != expected:
            errors.append(f"{partition} row_count expected {expected} got {actual}")

    summary = {
        "input_root": str(input_root),
        "files_total": len(files),
        "rows_total": sum(int(row["row_count"]) for row in source_rows),
        "file_counts": file_counts,
        "row_counts": row_counts,
        "checksum": bool(args.checksum),
        "errors": errors,
        "status": "passed" if not errors else "failed",
    }

    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.output_source_files:
        write_source_files(Path(args.output_source_files), source_rows)
        print(f"Wrote source file index: {args.output_source_files}")
    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
            handle.write("\n")
    if errors and bool(args.strict):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
