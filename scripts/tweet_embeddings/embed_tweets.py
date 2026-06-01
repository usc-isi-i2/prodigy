#!/usr/bin/env python3
"""Generate deterministic tweet embedding shards for the Ukraine-Russia corpus."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import math
import multiprocessing as mp
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any, Iterable
import unicodedata

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

try:
    import yaml
except ImportError:  # pragma: no cover - dependency is pinned in requirements.
    yaml = None


DEFAULT_INPUT_ROOT = "/dataMeR2/phil/data/ukr_rus_twitter/parquet"
DEFAULT_OUTPUT_ROOT = (
    "/dataMeR2/phil/data/ukr_rus_twitter/tweet_embeddings/"
    "gte-multilingual-base/version=v001"
)
DEFAULT_MODEL = "Alibaba-NLP/gte-multilingual-base"
DEFAULT_REVISION = "9bbca17d9273fd0d03d5725c7a4b0f6b45142062"
PREPROCESSING_VERSION = "tweet-text-v001"
OUTPUT_DIM = 768
URL_TOKEN = "<URL>"
USER_TOKEN = "<USER>"

TEXT_COLUMNS = [
    "tweetid",
    "text",
    "lang",
    "date",
    "userid",
    "tweet_type",
    "rt_tweetid",
    "rt_text",
    "qtd_tweetid",
    "qtd_text",
    "reply_statusid",
]

URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
HANDLE_RE = re.compile(r"(?<![\w@])@[\w_]+")
SPACE_RE = re.compile(r"\s+")
INVALID_TEXT_RE = re.compile(r"(?i)\b(deleted|unavailable|withheld)\b")

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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def coerce_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    text = str(value)
    if text in {"", "nan", "None", "<NA>"}:
        return ""
    return text


def has_value(value: Any) -> bool:
    return bool(coerce_text(value).strip())


def preprocess_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = URL_RE.sub(URL_TOKEN, text)
    text = HANDLE_RE.sub(USER_TOKEN, text)
    text = SPACE_RE.sub(" ", text).strip()
    return text


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def classify_tweet_type(value: Any) -> str:
    return coerce_text(value).strip().lower()


def assemble_text(row: dict[str, Any]) -> tuple[str, str]:
    """Choose one canonical content string for GNN-oriented tweet semantics."""
    tweet_type = classify_tweet_type(row.get("tweet_type"))
    own_text = coerce_text(row.get("text")).strip()
    rt_text = coerce_text(row.get("rt_text")).strip()
    qtd_text = coerce_text(row.get("qtd_text")).strip()

    is_quote = (
        "quote" in tweet_type
        or has_value(row.get("qtd_tweetid"))
        or bool(qtd_text)
    )
    is_retweet = (
        "retweet" in tweet_type
        or has_value(row.get("rt_tweetid"))
        or own_text.startswith("RT ")
    )

    if is_quote and own_text:
        return own_text, "own_text"
    if is_retweet and rt_text:
        return rt_text, "rt_text"
    if own_text:
        return own_text, "own_text"
    if rt_text:
        return rt_text, "rt_text"
    if qtd_text:
        return qtd_text, "qtd_text_fallback"
    return "", "empty"


def skip_reason_for_text(processed_text: str) -> str | None:
    if not processed_text:
        return "empty_after_preprocessing"
    if INVALID_TEXT_RE.search(processed_text):
        return "deleted_unavailable_withheld"
    return None


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_replace(tmp_path: Path, final_path: Path) -> None:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(tmp_path, final_path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp.{socket.gethostname()}.{os.getpid()}")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    atomic_replace(tmp_path, path)


def atomic_write_parquet(path: Path, rows: list[dict[str, Any]], schema: pa.Schema) -> str:
    tmp_path = path.with_name(f"{path.name}.tmp.{socket.gethostname()}.{os.getpid()}")
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, tmp_path, compression="zstd")
    atomic_replace(tmp_path, path)
    return sha256_file(path)


def atomic_write_npy(path: Path, array: np.ndarray) -> str:
    tmp_path = path.with_name(f"{path.name}.tmp.{socket.gethostname()}.{os.getpid()}")
    with tmp_path.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    atomic_replace(tmp_path, path)
    return sha256_file(path)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root(),
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip()


def package_versions() -> dict[str, str]:
    packages = [
        "numpy",
        "pandas",
        "pyarrow",
        "torch",
        "sentence-transformers",
        "transformers",
        "tokenizers",
        "huggingface-hub",
        "PyYAML",
    ]
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def torch_runtime_info() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {"torch_import_error": str(exc)}
    return {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_devices": [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
        if torch.cuda.is_available()
        else [],
    }


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


def setup_logger(output_root: Path, name: str) -> logging.Logger:
    log_dir = output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


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


def compute_token_lengths(model: Any, texts: list[str], max_seq_length: int) -> tuple[list[int], list[bool]]:
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        return [-1] * len(texts), [False] * len(texts)
    lengths: list[int] = []
    flags: list[bool] = []
    step = 2048
    for start in range(0, len(texts), step):
        batch = texts[start : start + step]
        try:
            encoded = tokenizer(
                batch,
                add_special_tokens=True,
                padding=False,
                truncation=False,
                return_length=True,
            )
            batch_lengths = encoded.get("length")
            if batch_lengths is None:
                batch_lengths = [len(ids) for ids in encoded["input_ids"]]
        except Exception:
            batch_lengths = [-1] * len(batch)
        lengths.extend(int(value) for value in batch_lengths)
        flags.extend(bool(value > max_seq_length) if value >= 0 else False for value in batch_lengths)
    return lengths, flags


def encode_texts(model: Any, texts: list[str], cfg: dict[str, Any]) -> np.ndarray:
    if not texts:
        return np.empty((0, int(cfg["embedding_dim"])), dtype=np.float16)
    chunks: list[np.ndarray] = []
    encode_batch_size = int(cfg["batch_size"])
    for start in range(0, len(texts), encode_batch_size):
        batch = texts[start : start + encode_batch_size]
        embeddings = model.encode(
            batch,
            batch_size=encode_batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        chunks.append(np.asarray(embeddings, dtype=np.float16))
    return np.vstack(chunks).astype(np.float16, copy=False)


def load_model_for_worker(rank: int, cfg: dict[str, Any]) -> tuple[Any, str]:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    from sentence_transformers import SentenceTransformer

    if bool(cfg["cpu"]):
        device = "cpu"
    elif torch.cuda.is_available():
        device = f"cuda:{rank}"
    else:
        device = "cpu"

    model = SentenceTransformer(
        cfg["model"],
        revision=cfg["revision"],
        trust_remote_code=True,
        device=device,
        cache_folder=cfg.get("cache_folder") or None,
    )
    model.max_seq_length = int(cfg["max_seq_length"])
    if bool(cfg["fp16"]) and device.startswith("cuda"):
        model = model.half()
    return model, device


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


def parse_shard_list(value: str) -> list[int] | None:
    if not value.strip():
        return None
    result: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            result.extend(range(int(left), int(right) + 1))
        else:
            result.append(int(part))
    return sorted(set(result))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Embed Ukraine-Russia tweet Parquet files into deterministic fp16 shards.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="", help="Optional YAML config to load before CLI overrides.")
    parser.add_argument("--input-root", default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--source-files", default="", help="Optional precomputed source_files.parquet.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--cache-folder", default="")
    parser.add_argument("--gpus", default="1,2,3,4", help="Recorded GPU list; also sets CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--worker-rank", type=int, default=-1, help="Run only one worker rank.")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--embedding-dim", type=int, default=OUTPUT_DIM)
    parser.add_argument("--shard-size", type=int, default=500_000)
    parser.add_argument("--parquet-batch-rows", type=int, default=65_536)
    parser.add_argument("--source-checksums", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rebuild-source-index", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smoke-shards", type=int, default=0, help="Limit to the first N selected shards.")
    parser.add_argument("--start-shard", type=int, default=0)
    parser.add_argument("--shards", default="", help="Comma/range list such as 0,4,8-12.")
    parser.add_argument("--index-only", action="store_true", help="Build source_files/config/manifest only.")
    parser.add_argument("--write-empty-skipped", action=argparse.BooleanOptionalAction, default=True)
    return parser


def parse_args() -> argparse.Namespace:
    base_parser = build_parser()
    config_only, _ = base_parser.parse_known_args()
    config_defaults: dict[str, Any] = {}
    if config_only.config:
        if yaml is None:
            raise RuntimeError("PyYAML is required to load --config")
        with Path(config_only.config).open("r", encoding="utf-8") as handle:
            config_defaults = yaml.safe_load(handle) or {}
        known_dests = {action.dest for action in base_parser._actions}
        config_defaults = {key: value for key, value in config_defaults.items() if key in known_dests}
        base_parser.set_defaults(**config_defaults)
    return base_parser.parse_args()


def args_to_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = vars(args).copy()
    cfg["input_root"] = str(Path(cfg["input_root"]))
    cfg["output_root"] = str(Path(cfg["output_root"]))
    cfg["preprocessing_version"] = PREPROCESSING_VERSION
    return cfg


def write_config(output_root: Path, cfg: dict[str, Any]) -> None:
    if yaml is None:
        atomic_write_json(output_root / "config.json", cfg)
        return
    tmp_path = output_root / f"config.yaml.tmp.{socket.gethostname()}.{os.getpid()}"
    with tmp_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=True)
    atomic_replace(tmp_path, output_root / "config.yaml")


def prepare_source_files(cfg: dict[str, Any], logger: logging.Logger) -> list[dict[str, Any]]:
    input_root = Path(cfg["input_root"])
    output_root = Path(cfg["output_root"])
    output_source_path = output_root / "source_files.parquet"
    external_source_path = Path(cfg["source_files"]) if cfg.get("source_files") else None

    if external_source_path and external_source_path.exists() and not bool(cfg["rebuild_source_index"]):
        logger.info("loading source index from %s", external_source_path)
        rows = load_source_index(external_source_path)
        if external_source_path.resolve() != output_source_path.resolve():
            shutil.copy2(external_source_path, output_source_path)
        return rows

    if output_source_path.exists() and not bool(cfg["rebuild_source_index"]):
        logger.info("loading source index from %s", output_source_path)
        return load_source_index(output_source_path)

    logger.info("building source index under %s checksums=%s", input_root, cfg["source_checksums"])
    rows = build_source_index(input_root, checksum=bool(cfg["source_checksums"]))
    write_source_index(output_source_path, rows)
    return rows


def selected_shards(cfg: dict[str, Any], source_rows: list[dict[str, Any]]) -> list[int]:
    total_rows = total_source_rows(source_rows)
    total_shards = math.ceil(total_rows / int(cfg["shard_size"]))
    explicit = parse_shard_list(str(cfg.get("shards") or ""))
    if explicit is not None:
        shards = [sid for sid in explicit if 0 <= sid < total_shards]
    else:
        shards = list(range(int(cfg["start_shard"]), total_shards))
    if int(cfg["smoke_shards"]) > 0:
        shards = shards[: int(cfg["smoke_shards"])]
    return shards


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


def main() -> int:
    args = parse_args()
    cfg = args_to_config(args)

    if cfg.get("gpus") and not bool(cfg["cpu"]):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["gpus"])

    output_root = Path(cfg["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    for child in ["shards", "skipped_rows", "logs"]:
        (output_root / child).mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_root, "run")
    write_config(output_root, cfg)
    source_rows = prepare_source_files(cfg, logger)
    selected = selected_shards(cfg, source_rows)
    logger.info(
        "source_files=%s source_rows=%s selected_shards=%s",
        len(source_rows),
        total_source_rows(source_rows),
        len(selected),
    )
    write_run_manifest(output_root, cfg, source_rows, selected)

    if bool(cfg["index_only"]):
        logger.info("index-only complete")
        return 0
    if not selected:
        logger.info("no shards selected")
        return 0

    if int(cfg["worker_rank"]) >= 0:
        worker_main(int(cfg["worker_rank"]), cfg, source_rows, selected)
    elif int(cfg["num_workers"]) == 1:
        worker_main(0, cfg, source_rows, selected)
    else:
        context = mp.get_context("spawn")
        processes = [
            context.Process(target=worker_main, args=(rank, cfg, source_rows, selected))
            for rank in range(int(cfg["num_workers"]))
        ]
        for process in processes:
            process.start()
        failed = False
        for process in processes:
            process.join()
            if process.exitcode != 0:
                failed = True
                logger.error("worker pid=%s exitcode=%s", process.pid, process.exitcode)
        if failed:
            raise RuntimeError("one or more workers failed")

    write_run_manifest(output_root, cfg, source_rows, selected)
    logger.info("embedding run complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
