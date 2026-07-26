"""Run-level manifest and embedding-index aggregation for bio embeddings."""

from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import socket
import sys
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.tweet_embeddings.io_utils import (
    atomic_replace,
    atomic_write_json,
    atomic_write_parquet,
    git_commit,
    package_versions,
    sha256_file,
    torch_runtime_info,
    utc_now,
)

from .constants import PREPROCESSING_VERSION
from .schemas import BIO_EMBEDDING_INDEX_SCHEMA


def read_shard_manifests(output_root: Path) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    for path in sorted((output_root / "shards").glob("shard-*.manifest.json")):
        with path.open("r", encoding="utf-8") as handle:
            manifests.append(json.load(handle))
    return manifests


def _package_versions_with_duckdb() -> dict[str, str]:
    versions = package_versions()
    try:
        versions["duckdb"] = importlib.metadata.version("duckdb")
    except importlib.metadata.PackageNotFoundError:
        versions["duckdb"] = "not-installed"
    return versions


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_embedding_index(output_root: Path, shard_manifests: list[dict[str, Any]]) -> dict[str, Any]:
    index_path = output_root / "bio_embedding_index.parquet"
    tmp_path = index_path.with_name(
        f"{index_path.name}.tmp.{socket.gethostname()}.{os.getpid()}"
    )
    if tmp_path.exists():
        tmp_path.unlink()
    index_path.parent.mkdir(parents=True, exist_ok=True)

    writer = pq.ParquetWriter(tmp_path, BIO_EMBEDDING_INDEX_SCHEMA, compression="zstd")
    rows_written = 0
    try:
        for manifest in sorted(shard_manifests, key=lambda row: int(row["shard_id"])):
            meta_path = output_root / str(manifest["metadata_path"])
            if not meta_path.exists():
                continue
            meta = pq.read_table(meta_path, columns=["bio_id", "bio_hash", "embedding_row"])
            n_rows = meta.num_rows
            if n_rows == 0:
                continue
            table = pa.Table.from_arrays(
                [
                    meta.column("bio_id"),
                    meta.column("bio_hash"),
                    pa.array([str(manifest["embedding_path"])] * n_rows, type=pa.string()),
                    meta.column("embedding_row"),
                    pa.array([int(manifest["embedding_dim"])] * n_rows, type=pa.int32()),
                    pa.array([str(manifest["embedding_dtype"])] * n_rows, type=pa.string()),
                    pa.array([str(manifest["model"])] * n_rows, type=pa.string()),
                    pa.array([str(manifest["revision"])] * n_rows, type=pa.string()),
                ],
                schema=BIO_EMBEDDING_INDEX_SCHEMA,
            )
            writer.write_table(table)
            rows_written += n_rows
    finally:
        writer.close()

    atomic_replace(tmp_path, index_path)
    return {
        "path": "bio_embedding_index.parquet",
        "rows": int(rows_written),
        "sha256": sha256_file(index_path),
    }


def _write_manifest_table(output_root: Path, shard_manifests: list[dict[str, Any]]) -> None:
    manifest_table_rows = [
        {
            "shard_id": int(row["shard_id"]),
            "bio_id_start": int(row["bio_id_start"]),
            "bio_id_end": int(row["bio_id_end"]),
            "bio_rows": int(row["bio_rows"]),
            "embedding_path": str(row["embedding_path"]),
            "metadata_path": str(row["metadata_path"]),
            "embedding_sha256": str(row["embedding_sha256"]),
            "metadata_sha256": str(row["metadata_sha256"]),
            "validation_status": str(row.get("validation_status", "")),
            "completed_at": str(row.get("completed_at", "")),
        }
        for row in shard_manifests
    ]
    manifest_schema = pa.schema(
        [
            ("shard_id", pa.int32()),
            ("bio_id_start", pa.int64()),
            ("bio_id_end", pa.int64()),
            ("bio_rows", pa.int64()),
            ("embedding_path", pa.string()),
            ("metadata_path", pa.string()),
            ("embedding_sha256", pa.string()),
            ("metadata_sha256", pa.string()),
            ("validation_status", pa.string()),
            ("completed_at", pa.string()),
        ]
    )
    atomic_write_parquet(output_root / "manifest.parquet", manifest_table_rows, manifest_schema)


def write_run_manifest(
    output_root: Path,
    cfg: dict[str, Any],
    source_rows: list[dict[str, Any]],
    selected: list[int],
) -> None:
    shard_manifests = read_shard_manifests(output_root)
    _write_manifest_table(output_root, shard_manifests)
    embedding_index = None
    if shard_manifests:
        embedding_index = write_embedding_index(output_root, shard_manifests)

    bio_texts_path = output_root / "bio_texts.parquet"
    source_files_path = output_root / "source_files.parquet"
    bio_index_summary = load_json_if_exists(output_root / "bio_index_summary.json")
    bio_text_rows = (
        int(pq.ParquetFile(bio_texts_path).metadata.num_rows)
        if bio_texts_path.exists()
        else 0
    )
    source_fingerprint = sha256_file(source_files_path) if source_files_path.exists() else ""
    bio_texts_sha = sha256_file(bio_texts_path) if bio_texts_path.exists() else ""

    payload = {
        "created_at": utc_now(),
        "git_commit": git_commit(),
        "command": " ".join(sys.argv),
        "hostname": socket.gethostname(),
        "python": sys.version,
        "package_versions": _package_versions_with_duckdb(),
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
            "rows": sum(int(row["row_count"]) for row in source_rows),
            "source_files_path": "source_files.parquet",
            "source_fingerprint_sha256": source_fingerprint,
        },
        "bio_index": {
            "bio_texts_path": "bio_texts.parquet" if bio_texts_path.exists() else "",
            "bio_text_rows": bio_text_rows,
            "bio_texts_sha256": bio_texts_sha,
            "summary_path": "bio_index_summary.json"
            if (output_root / "bio_index_summary.json").exists()
            else "",
            "summary": bio_index_summary,
        },
        "selected_shards": selected,
        "completed_shards": len(shard_manifests),
        "embedded_rows": sum(int(row["bio_rows"]) for row in shard_manifests),
        "embedding_index": embedding_index,
        "shards": shard_manifests,
    }
    atomic_write_json(output_root / "manifest.json", payload)
