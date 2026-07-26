#!/usr/bin/env python3
"""Validate bio embedding index and shard outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pyarrow.parquet as pq

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.tweet_embeddings.io_utils import atomic_write_json, sha256_file

from scripts.bio_embeddings.constants import DEFAULT_MODEL, DEFAULT_OUTPUT_ROOT, DEFAULT_REVISION


def _require_duckdb() -> Any:
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - dependency is pinned for runtime.
        raise RuntimeError(
            "duckdb is required for full bio embedding validation. Install "
            "scripts/bio_embeddings/requirements-embeddings.txt."
        ) from exc
    return duckdb


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def shard_paths(output_root: Path, shard_id: int) -> dict[str, Path]:
    stem = f"shard-{shard_id:06d}"
    return {
        "emb": output_root / "shards" / f"{stem}.emb.npy",
        "meta": output_root / "shards" / f"{stem}.meta.parquet",
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
    meta_table = pq.read_table(paths["meta"], columns=["bio_id", "bio_hash", "embedding_row"])
    meta_rows = meta_table.num_rows
    result.update(
        {
            "embedding_shape": [int(x) for x in array.shape],
            "embedding_dtype": str(array.dtype),
            "metadata_rows": int(meta_rows),
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
    if int(meta_rows) > 0:
        embedding_rows = np.asarray(meta_table.column("embedding_row").to_pylist(), dtype=np.int64)
        expected_rows = np.arange(int(meta_rows), dtype=np.int64)
        if not np.array_equal(embedding_rows, expected_rows):
            result["errors"].append("metadata_embedding_row_not_contiguous")
    if bool(args.check_finite) and not np.isfinite(array).all():
        result["errors"].append("non_finite_embedding")

    if bool(args.check_checksums):
        emb_sha = sha256_file(paths["emb"])
        meta_sha = sha256_file(paths["meta"])
        if manifest.get("embedding_sha256") != emb_sha:
            result["errors"].append("embedding_checksum_mismatch")
        if manifest.get("metadata_sha256") != meta_sha:
            result["errors"].append("metadata_checksum_mismatch")

    if str(manifest.get("model")) != args.expected_model:
        result["errors"].append("wrong_model")
    if str(manifest.get("revision")) != args.expected_revision:
        result["errors"].append("wrong_revision")

    if bool(args.check_norms):
        norm_result = check_norms(array, int(args.norm_sample), float(args.norm_tolerance))
        result["norms"] = norm_result
        if not norm_result["passed"]:
            result["errors"].append("norm_tolerance_failed")

    result["status"] = "passed" if not result["errors"] else "failed"
    return result


def validate_global(output_root: Path, selected: list[int] | None, embedded_rows: int) -> dict[str, Any]:
    duckdb = _require_duckdb()
    bio_texts = output_root / "bio_texts.parquet"
    embedding_index = output_root / "bio_embedding_index.parquet"
    result: dict[str, Any] = {"errors": []}
    if not bio_texts.exists():
        result["errors"].append("missing_bio_texts")
    if not embedding_index.exists():
        result["errors"].append("missing_bio_embedding_index")
    if result["errors"]:
        result["status"] = "failed"
        return result

    conn = duckdb.connect()
    try:
        bio_rows, duplicate_hashes, duplicate_bio_ids = conn.execute(
            """
            SELECT
                count(*) AS bio_rows,
                count(*) - count(DISTINCT bio_hash) AS duplicate_hashes,
                count(*) - count(DISTINCT bio_id) AS duplicate_bio_ids
            FROM read_parquet(?)
            """,
            [str(bio_texts)],
        ).fetchone()
        index_rows, duplicate_index_bio_ids, wrong_dim, wrong_dtype = conn.execute(
            """
            SELECT
                count(*) AS index_rows,
                count(*) - count(DISTINCT bio_id) AS duplicate_index_bio_ids,
                sum(CASE WHEN embedding_dim != 768 THEN 1 ELSE 0 END) AS wrong_dim,
                sum(CASE WHEN embedding_dtype != 'float16' THEN 1 ELSE 0 END) AS wrong_dtype
            FROM read_parquet(?)
            """,
            [str(embedding_index)],
        ).fetchone()
        result.update(
            {
                "bio_text_rows": int(bio_rows),
                "embedding_index_rows": int(index_rows),
                "duplicate_bio_hashes": int(duplicate_hashes),
                "duplicate_bio_ids": int(duplicate_bio_ids),
                "duplicate_index_bio_ids": int(duplicate_index_bio_ids),
            }
        )
        if int(duplicate_hashes) != 0:
            result["errors"].append("duplicate_bio_hash")
        if int(duplicate_bio_ids) != 0:
            result["errors"].append("duplicate_bio_id")
        if int(duplicate_index_bio_ids) != 0:
            result["errors"].append("duplicate_index_bio_id")
        if int(wrong_dim or 0) != 0:
            result["errors"].append("index_wrong_dim")
        if int(wrong_dtype or 0) != 0:
            result["errors"].append("index_wrong_dtype")
        if int(index_rows) != int(embedded_rows):
            result["errors"].append("index_embedded_row_mismatch")

        if selected is None:
            if int(embedded_rows) != int(bio_rows):
                result["errors"].append("embedded_bio_text_row_mismatch")
            missing = conn.execute(
                """
                SELECT count(*)
                FROM read_parquet(?) AS b
                LEFT JOIN read_parquet(?) AS i
                    ON b.bio_id = i.bio_id
                    AND b.bio_hash = i.bio_hash
                WHERE i.bio_id IS NULL
                """,
                [str(bio_texts), str(embedding_index)],
            ).fetchone()[0]
            result["missing_index_rows"] = int(missing)
            if int(missing) != 0:
                result["errors"].append("embedding_index_missing_bio_text_rows")
    finally:
        conn.close()

    result["status"] = "passed" if not result["errors"] else "failed"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate bio embedding shards.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--shards", default="", help="Comma/range list such as 0,4,8-12.")
    parser.add_argument("--expected-dim", type=int, default=768)
    parser.add_argument("--expected-dtype", default="float16")
    parser.add_argument("--expected-model", default=DEFAULT_MODEL)
    parser.add_argument("--expected-revision", default=DEFAULT_REVISION)
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
    global_result = validate_global(output_root, selected, embedded_rows)
    summary = {
        "output_root": str(output_root),
        "validated_shards": len(results),
        "embedded_rows": embedded_rows,
        "failed_shards": [result["shard_id"] for result in results if result["status"] != "passed"],
        "global": global_result,
        "results": results,
    }
    summary["status"] = (
        "passed"
        if not summary["failed_shards"] and global_result["status"] == "passed"
        else "failed"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.summary_json:
        atomic_write_json(Path(args.summary_json), summary)
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
