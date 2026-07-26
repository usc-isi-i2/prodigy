"""Run-level manifest aggregation."""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import sys
from typing import Any

import pyarrow as pa

from .constants import PREPROCESSING_VERSION
from .io_utils import (
    atomic_write_json,
    atomic_write_parquet,
    git_commit,
    package_versions,
    sha256_file,
    torch_runtime_info,
    utc_now,
)
from .source_index import total_source_rows


def read_shard_manifests(output_root: Path) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    for path in sorted((output_root / "shards").glob("shard-*.manifest.json")):
        with path.open("r", encoding="utf-8") as handle:
            manifests.append(json.load(handle))
    return manifests


def write_run_manifest(
    output_root: Path,
    cfg: dict[str, Any],
    source_rows: list[dict[str, Any]],
    selected: list[int],
) -> None:
    shard_manifests = read_shard_manifests(output_root)
    manifest_table_rows = [
        {
            "shard_id": int(row["shard_id"]),
            "source_global_row_start": int(row["source_global_row_start"]),
            "source_global_row_end": int(row["source_global_row_end"]),
            "source_rows": int(row["source_rows"]),
            "embedded_rows": int(row["embedded_rows"]),
            "skipped_rows": int(row["skipped_rows"]),
            "embedding_path": str(row["embedding_path"]),
            "metadata_path": str(row["metadata_path"]),
            "skipped_path": str(row.get("skipped_path", "")),
            "embedding_sha256": str(row["embedding_sha256"]),
            "metadata_sha256": str(row["metadata_sha256"]),
            "skipped_sha256": str(row.get("skipped_sha256", "")),
            "validation_status": str(row.get("validation_status", "")),
            "completed_at": str(row.get("completed_at", "")),
        }
        for row in shard_manifests
    ]
    manifest_schema = pa.schema(
        [
            ("shard_id", pa.int32()),
            ("source_global_row_start", pa.int64()),
            ("source_global_row_end", pa.int64()),
            ("source_rows", pa.int64()),
            ("embedded_rows", pa.int64()),
            ("skipped_rows", pa.int64()),
            ("embedding_path", pa.string()),
            ("metadata_path", pa.string()),
            ("skipped_path", pa.string()),
            ("embedding_sha256", pa.string()),
            ("metadata_sha256", pa.string()),
            ("skipped_sha256", pa.string()),
            ("validation_status", pa.string()),
            ("completed_at", pa.string()),
        ]
    )
    atomic_write_parquet(output_root / "manifest.parquet", manifest_table_rows, manifest_schema)

    total_rows = total_source_rows(source_rows)
    payload = {
        "created_at": utc_now(),
        "git_commit": git_commit(),
        "command": " ".join(sys.argv),
        "hostname": socket.gethostname(),
        "python": sys.version,
        "package_versions": package_versions(),
        "torch_runtime": torch_runtime_info(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "config": cfg,
        "model": {
            "id": cfg["model"],
            "revision": cfg["revision"],
            "trust_remote_code": True,
            "pooling": "SentenceTransformers CLS/pooling config",
            "max_seq_length": int(cfg["max_seq_length"]),
            "embedding_dim": int(cfg["embedding_dim"]),
            "dtype": "float16",
            "normalize_embeddings": True,
        },
        "preprocessing_version": PREPROCESSING_VERSION,
        "source": {
            "input_root": cfg["input_root"],
            "files": len(source_rows),
            "rows": total_rows,
            "source_files_path": "source_files.parquet",
            "source_fingerprint_sha256": sha256_file(output_root / "source_files.parquet"),
        },
        "selected_shards": selected,
        "completed_shards": len(shard_manifests),
        "embedded_rows": sum(int(row["embedded_rows"]) for row in shard_manifests),
        "skipped_rows": sum(int(row["skipped_rows"]) for row in shard_manifests),
        "shards": shard_manifests,
    }
    atomic_write_json(output_root / "manifest.json", payload)
