#!/usr/bin/env python3
"""Validate tweet embedding shard outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


DEFAULT_OUTPUT_ROOT = (
    "/dataMeR2/phil/data/ukr_rus_twitter/tweet_embeddings/"
    "gte-multilingual-base/version=v001"
)


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def shard_paths(output_root: Path, shard_id: int) -> dict[str, Path]:
    stem = f"shard-{shard_id:06d}"
    return {
        "emb": output_root / "shards" / f"{stem}.emb.npy",
        "meta": output_root / "shards" / f"{stem}.meta.parquet",
        "skip": output_root / "skipped_rows" / f"{stem}.skipped.parquet",
        "manifest": output_root / "shards" / f"{stem}.manifest.json",
    }


def discover_shard_ids(output_root: Path) -> list[int]:
    shard_ids: list[int] = []
    for path in sorted((output_root / "shards").glob("shard-*.emb.npy")):
        try:
            shard_ids.append(int(path.name.split(".")[0].split("-")[1]))
        except Exception:
            continue
    return shard_ids


def parse_shard_list(value: str) -> list[int] | None:
    if not value.strip():
        return None
    shard_ids: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            shard_ids.extend(range(int(left), int(right) + 1))
        else:
            shard_ids.append(int(part))
    return sorted(set(shard_ids))


def check_norms(array: np.ndarray, sample: int, tolerance: float) -> dict[str, Any]:
    if array.shape[0] == 0:
        return {"checked": 0, "max_abs_error": 0.0, "passed": True}
    if sample > 0 and array.shape[0] > sample:
        idx = np.linspace(0, array.shape[0] - 1, num=sample, dtype=np.int64)
        values = np.asarray(array[idx], dtype=np.float32)
    else:
        values = np.asarray(array, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1)
    max_abs_error = float(np.max(np.abs(norms - 1.0))) if norms.size else 0.0
    return {
        "checked": int(values.shape[0]),
        "max_abs_error": max_abs_error,
        "passed": bool(max_abs_error <= tolerance),
    }


def validate_one(output_root: Path, shard_id: int, args: argparse.Namespace) -> dict[str, Any]:
    paths = shard_paths(output_root, shard_id)
    result: dict[str, Any] = {"shard_id": shard_id, "errors": []}
    for key in ("emb", "meta", "manifest"):
        if not paths[key].exists():
            result["errors"].append(f"missing_{key}")
    if result["errors"]:
        result["status"] = "failed"
        return result

    manifest = load_json(paths["manifest"])
    array = np.load(paths["emb"], mmap_mode="r")
    meta_rows = pq.ParquetFile(paths["meta"]).metadata.num_rows
    skipped_rows = pq.ParquetFile(paths["skip"]).metadata.num_rows if paths["skip"].exists() else 0

    result.update(
        {
            "embedding_shape": [int(x) for x in array.shape],
            "embedding_dtype": str(array.dtype),
            "metadata_rows": int(meta_rows),
            "skipped_rows": int(skipped_rows),
        }
    )
    if array.ndim != 2:
        result["errors"].append(f"wrong_ndim:{array.ndim}")
    elif array.shape[1] != int(args.expected_dim):
        result["errors"].append(f"wrong_dim:{array.shape[1]}")
    if str(array.dtype) != args.expected_dtype:
        result["errors"].append(f"wrong_dtype:{array.dtype}")
    if int(meta_rows) != int(array.shape[0]):
        result["errors"].append("metadata_embedding_row_mismatch")
    if bool(args.check_finite) and not np.isfinite(array).all():
        result["errors"].append("non_finite_embedding")

    if bool(args.check_checksums):
        emb_sha = sha256_file(paths["emb"])
        meta_sha = sha256_file(paths["meta"])
        if manifest.get("embedding_sha256") != emb_sha:
            result["errors"].append("embedding_checksum_mismatch")
        if manifest.get("metadata_sha256") != meta_sha:
            result["errors"].append("metadata_checksum_mismatch")
        if paths["skip"].exists() and manifest.get("skipped_sha256", "") != sha256_file(paths["skip"]):
            result["errors"].append("skipped_checksum_mismatch")

    if bool(args.check_norms):
        norm_result = check_norms(array, int(args.norm_sample), float(args.norm_tolerance))
        result["norms"] = norm_result
        if not norm_result["passed"]:
            result["errors"].append("norm_tolerance_failed")

    result["status"] = "passed" if not result["errors"] else "failed"
    return result


def source_total_rows(output_root: Path) -> int | None:
    path = output_root / "source_files.parquet"
    if not path.exists():
        return None
    table = pq.read_table(path, columns=["row_count"])
    return int(sum(table.column("row_count").to_pylist()))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate tweet embedding shards.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--shards", default="", help="Comma/range list such as 0,4,8-12.")
    parser.add_argument("--expected-dim", type=int, default=768)
    parser.add_argument("--expected-dtype", default="float16")
    parser.add_argument("--check-finite", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--check-checksums", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--check-norms", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--norm-sample", type=int, default=10000, help="0 checks every row.")
    parser.add_argument("--norm-tolerance", type=float, default=2e-2)
    parser.add_argument("--summary-json", default="")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    selected = parse_shard_list(args.shards)
    shard_ids = selected if selected is not None else discover_shard_ids(output_root)
    if not shard_ids:
        raise FileNotFoundError(f"No embedding shards found under {output_root / 'shards'}")

    results = [validate_one(output_root, shard_id, args) for shard_id in shard_ids]
    embedded_rows = sum(int(result.get("metadata_rows", 0)) for result in results)
    skipped_rows = sum(int(result.get("skipped_rows", 0)) for result in results)
    source_rows = source_total_rows(output_root)
    summary = {
        "output_root": str(output_root),
        "validated_shards": len(results),
        "embedded_rows": embedded_rows,
        "skipped_rows": skipped_rows,
        "source_rows": source_rows,
        "complete_source_coverage": (
            bool(source_rows is not None and embedded_rows + skipped_rows == source_rows)
            if selected is None
            else None
        ),
        "failed_shards": [result["shard_id"] for result in results if result["status"] != "passed"],
        "results": results,
    }
    coverage_failed = (
        selected is None
        and source_rows is not None
        and embedded_rows + skipped_rows != source_rows
    )
    summary["coverage_error"] = bool(coverage_failed)
    summary["status"] = "passed" if not summary["failed_shards"] and not coverage_failed else "failed"
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
            handle.write("\n")
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
