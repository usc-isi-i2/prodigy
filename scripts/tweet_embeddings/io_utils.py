"""Runtime metadata, checksums, and atomic file writes."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
