"""Worker orchestration for shard embedding and validation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import time
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from .constants import PREPROCESSING_VERSION
from .io_utils import atomic_write_json, atomic_write_npy, atomic_write_parquet, sha256_file, utc_now
from .logging_utils import setup_logger
from .model_backend import compute_token_lengths, encode_texts, load_model_for_worker
from .reader import prepare_shard_texts
from .schemas import META_SCHEMA, SKIP_SCHEMA
from .source_index import total_source_rows


def shard_paths(output_root: Path, shard_id: int) -> dict[str, Path]:
    stem = f"shard-{shard_id:06d}"
    return {
        "emb": output_root / "shards" / f"{stem}.emb.npy",
        "meta": output_root / "shards" / f"{stem}.meta.parquet",
        "skip": output_root / "skipped_rows" / f"{stem}.skipped.parquet",
        "manifest": output_root / "shards" / f"{stem}.manifest.json",
    }


def validate_basic_shard(paths: dict[str, Path], expected_dim: int) -> tuple[bool, str]:
    if not paths["emb"].exists() or not paths["meta"].exists() or not paths["manifest"].exists():
        return False, "missing_required_file"
    try:
        with paths["manifest"].open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("embedding_sha256") != sha256_file(paths["emb"]):
            return False, "embedding_checksum_mismatch"
        if manifest.get("metadata_sha256") != sha256_file(paths["meta"]):
            return False, "metadata_checksum_mismatch"
        if paths["skip"].exists() and manifest.get("skipped_sha256", "") != sha256_file(paths["skip"]):
            return False, "skipped_checksum_mismatch"
        embeddings = np.load(paths["emb"], mmap_mode="r")
        if embeddings.dtype != np.float16:
            return False, f"wrong_dtype_{embeddings.dtype}"
        if embeddings.ndim != 2 or embeddings.shape[1] != expected_dim:
            return False, f"wrong_shape_{embeddings.shape}"
        meta_rows = pq.ParquetFile(paths["meta"]).metadata.num_rows
        if int(meta_rows) != int(embeddings.shape[0]):
            return False, "metadata_row_count_mismatch"
    except Exception as exc:
        return False, f"validation_error:{exc}"
    return True, "ok"


def write_shard(
    rank: int,
    shard_id: int,
    model: Any,
    device: str,
    input_root: Path,
    output_root: Path,
    source_rows: list[dict[str, Any]],
    cfg: dict[str, Any],
    logger: logging.Logger,
) -> dict[str, Any]:
    paths = shard_paths(output_root, shard_id)
    valid, reason = validate_basic_shard(paths, int(cfg["embedding_dim"]))
    if valid and bool(cfg["resume"]):
        logger.info("rank=%s shard=%06d skip complete", rank, shard_id)
        with paths["manifest"].open("r", encoding="utf-8") as handle:
            return json.load(handle)
    if reason != "missing_required_file" and bool(cfg["resume"]):
        logger.info("rank=%s shard=%06d recompute because %s", rank, shard_id, reason)

    start_time = time.time()
    logger.info("rank=%s shard=%06d start", rank, shard_id)
    texts, metadata_rows, skipped_rows = prepare_shard_texts(input_root, source_rows, shard_id, cfg)
    lengths, truncation_flags = compute_token_lengths(model, texts, int(cfg["max_seq_length"]))
    for row, length, flag in zip(metadata_rows, lengths, truncation_flags):
        row["token_length"] = int(length)
        row["truncation_flag"] = bool(flag)

    embeddings = encode_texts(model, texts, cfg)
    if embeddings.ndim != 2 or embeddings.shape[1] != int(cfg["embedding_dim"]):
        raise RuntimeError(f"shard {shard_id} wrong embedding shape: {embeddings.shape}")
    if embeddings.dtype != np.float16:
        raise RuntimeError(f"shard {shard_id} wrong embedding dtype: {embeddings.dtype}")
    if not np.isfinite(embeddings).all():
        raise RuntimeError(f"shard {shard_id} contains non-finite embeddings")
    if len(metadata_rows) != embeddings.shape[0]:
        raise RuntimeError(f"shard {shard_id} metadata/embedding row mismatch")

    emb_sha = atomic_write_npy(paths["emb"], embeddings)
    meta_sha = atomic_write_parquet(paths["meta"], metadata_rows, META_SCHEMA)
    skipped_sha = ""
    if skipped_rows or bool(cfg["write_empty_skipped"]):
        skipped_sha = atomic_write_parquet(paths["skip"], skipped_rows, SKIP_SCHEMA)
    elif paths["skip"].exists():
        paths["skip"].unlink()

    duration = time.time() - start_time
    source_start = shard_id * int(cfg["shard_size"])
    source_end = min(source_start + int(cfg["shard_size"]), total_source_rows(source_rows))
    manifest = {
        "shard_id": shard_id,
        "worker_rank": rank,
        "device": device,
        "source_global_row_start": int(source_start),
        "source_global_row_end": int(source_end),
        "source_rows": int(source_end - source_start),
        "embedded_rows": int(embeddings.shape[0]),
        "skipped_rows": int(len(skipped_rows)),
        "embedding_path": paths["emb"].relative_to(output_root).as_posix(),
        "metadata_path": paths["meta"].relative_to(output_root).as_posix(),
        "skipped_path": paths["skip"].relative_to(output_root).as_posix() if paths["skip"].exists() else "",
        "embedding_sha256": emb_sha,
        "metadata_sha256": meta_sha,
        "skipped_sha256": skipped_sha,
        "embedding_dtype": "float16",
        "embedding_dim": int(cfg["embedding_dim"]),
        "model": cfg["model"],
        "revision": cfg["revision"],
        "preprocessing_version": PREPROCESSING_VERSION,
        "duration_seconds": duration,
        "rows_per_second": float((source_end - source_start) / duration) if duration > 0 else 0.0,
        "completed_at": utc_now(),
        "validation_status": "passed",
    }
    atomic_write_json(paths["manifest"], manifest)
    logger.info(
        "rank=%s shard=%06d done embedded=%s skipped=%s %.1fs",
        rank,
        shard_id,
        embeddings.shape[0],
        len(skipped_rows),
        duration,
    )
    return manifest


def worker_main(rank: int, cfg: dict[str, Any], source_rows: list[dict[str, Any]], shard_ids: list[int]) -> None:
    output_root = Path(cfg["output_root"])
    input_root = Path(cfg["input_root"])
    logger = setup_logger(output_root, f"worker-{rank}")
    model, device = load_model_for_worker(rank, cfg)
    owned = [sid for sid in shard_ids if sid % int(cfg["num_workers"]) == rank]
    logger.info("rank=%s device=%s owned_shards=%s", rank, device, len(owned))
    for shard_id in owned:
        write_shard(rank, shard_id, model, device, input_root, output_root, source_rows, cfg, logger)
