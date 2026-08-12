#!/usr/bin/env python3
"""Audit aggregate identity and biography overlap without emitting private rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


TWITTER_DATASETS = [
    "ukraine",
    "covid",
    "midterm",
    "covid-political",
    "ukraine-suspended",
    "election2020-political",
    "twibot20",
    "hongkong",
]
ALL_DATASETS = TWITTER_DATASETS + ["facebook-page-reference"]
TABLE_KEY = {name: name.replace("-", "_") for name in ALL_DATASETS}

GRAPH_NODES = {
    "ukraine": 10_400_775,
    "covid": 23_012_850,
    "midterm": 341_908,
    "covid-political": 78_672,
    "ukraine-suspended": 72_295,
    "election2020-political": 78_932,
    "twibot20": 162_990,
    "hongkong": 333_800,
    "facebook-page-reference": 150_000,
}

EXACT_ID_SCOPE = {
    "ukraine": "full_graph_global_twitter_id",
    "covid": "full_graph_global_twitter_id",
    "midterm": "full_graph_global_twitter_id",
    "hongkong": "not_measurable_hashed_dataset_namespace",
    "ukraine-suspended": "partial_global_twitter_id_array",
    "covid-political": "not_measurable_row_indices_only",
    "election2020-political": "not_measurable_row_indices_only",
    "twibot20": "not_measurable_dataset_internal_namespace",
    "facebook-page-reference": "incompatible_platform",
}

RAW_GRAPH_SPECS = {
    "ukraine": {
        "module": "generate_ukr_rus_retweet_graph_from_parquet",
        "parquet_root": "ukr_rus_twitter/parquet",
        "bio_root": "ukr_rus_twitter/bio_embeddings/gte-multilingual-base/version=v001",
    },
    "covid": {
        "module": "generate_covid19_twitter_retweet_graph_from_parquet",
        "parquet_root": "covid19_twitter/parquet",
        "bio_root": "covid19_twitter/bio_embeddings/gte-multilingual-base/version=v001",
    },
    "midterm": {
        "module": "generate_midterm_retweet_graph_from_parquet",
        "parquet_root": "midterm/parquet",
        "bio_root": "midterm/bio_embeddings/gte-multilingual-base/version=v001",
    },
}

SOCIAL_EMBEDDINGS = {
    "covid-political": "covid_political/embeddings/user_bio_embeddings_gte_multilingual_base.pt",
    "ukraine-suspended": "ukr_rus_suspended/embeddings/user_bio_embeddings_gte_multilingual_base.pt",
    "election2020-political": "election2020/embeddings/user_bio_embeddings_gte_multilingual_base.pt",
}

SOCIAL_CSVS = {
    "covid-political": "social_llm_data/covid/user_data.csv",
    "ukraine-suspended": "social_llm_data/ukr_rus_suspended/user_data.csv",
    "election2020-political": "social_llm_data/election2020/user_data.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="/dataMeR1/phil/data")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--memory-limit", default="80GB")
    parser.add_argument("--threads", type=int, default=24)
    return parser.parse_args()


def sha256_file(path: Path, block_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def source_manifest_digest(paths: Iterable[Path]) -> str:
    """Hash relative source names and sizes, not source contents or identities."""
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: str(item)):
        stat = path.stat()
        digest.update(f"{path}\t{stat.st_size}\n".encode())
    return digest.hexdigest()


def pair_status(scope_a: str, scope_b: str) -> tuple[str, str]:
    if "incompatible_platform" in {scope_a, scope_b}:
        return "incompatible_platform", "different platform identifier namespaces"
    unavailable = [scope for scope in (scope_a, scope_b) if scope.startswith("not_measurable")]
    if unavailable:
        return "not_measurable", "; ".join(sorted(set(unavailable)))
    if "partial_global_twitter_id_array" in {scope_a, scope_b}:
        return "partial_exact", "Ukraine-Suspended stable IDs cover only a graph subset"
    return "exact", "comparable platform-global Twitter user IDs"


def overlap_metrics(size_a: int, size_b: int, intersection: int) -> dict[str, float]:
    union = size_a + size_b - intersection
    return {
        "fraction_a": intersection / size_a if size_a else 0.0,
        "fraction_b": intersection / size_b if size_b else 0.0,
        "fraction_smaller": intersection / min(size_a, size_b) if min(size_a, size_b) else 0.0,
        "jaccard": intersection / union if union else 0.0,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def import_graph_builder(module_name: str):
    from importlib import import_module

    return import_module(f"scripts.graph_construction.{module_name}")


def configure_duckdb(conn: Any, output_dir: Path, memory_limit: str, threads: int) -> None:
    temp_dir = output_dir / "duckdb_tmp"
    temp_dir.mkdir(parents=True, exist_ok=False)
    conn.execute("SET temp_directory=?", [str(temp_dir)])
    conn.execute("SET memory_limit=?", [memory_limit])
    conn.execute(f"SET threads={int(threads)}")


def create_raw_graph_tables(conn: Any, data_root: Path, name: str, spec: dict[str, str]) -> dict[str, Any]:
    module = import_graph_builder(spec["module"])
    parquet_root = data_root / spec["parquet_root"]
    files = module.resolve_input_files(str(parquet_root), [], 0)
    module._build_source_scan(conn, files)  # exact construction policy
    module._build_retweet_events(conn, "")
    key = TABLE_KEY[name]
    conn.execute(
        f"""
        CREATE TABLE ids_{key} AS
        SELECT DISTINCT CAST(userid AS VARCHAR) AS id FROM retweet_events
        UNION
        SELECT DISTINCT CAST(rt_userid AS VARCHAR) AS id FROM retweet_events
        """
    )
    window = conn.execute(
        "SELECT min(observed_at), max(observed_at) FROM retweet_events"
    ).fetchone()

    bio_root = data_root / spec["bio_root"]
    observations = bio_root / "user_bio_observations.parquet"
    bio_texts = bio_root / "bio_texts.parquet"
    conn.execute(
        f"""
        CREATE TABLE bios_{key} AS
        WITH ranked AS (
            SELECT
                CAST(o.userid AS VARCHAR) AS id,
                CAST(o.bio_hash AS VARCHAR) AS bio_hash,
                row_number() OVER (
                    PARTITION BY CAST(o.userid AS VARCHAR)
                    ORDER BY COALESCE(o.last_seen_at, o.first_seen_at) DESC NULLS LAST,
                             o.bio_hash DESC
                ) AS rn
            FROM read_parquet(?) AS o
            INNER JOIN ids_{key} AS i ON CAST(o.userid AS VARCHAR) = i.id
            WHERE o.bio_hash IS NOT NULL AND trim(CAST(o.bio_hash AS VARCHAR)) <> ''
        )
        SELECT r.id, r.bio_hash, length(t.normalized_bio_text)::BIGINT AS bio_len
        FROM ranked AS r
        LEFT JOIN read_parquet(?) AS t USING (bio_hash)
        WHERE r.rn = 1
        """,
        [str(observations), str(bio_texts)],
    )
    return {
        "source_files": len(files),
        "source_manifest_sha256": source_manifest_digest(Path(path) for path in files),
        "window_start": window[0].isoformat() if window[0] else "",
        "window_end": window[1].isoformat() if window[1] else "",
    }


def register_arrow_table(conn: Any, name: str, columns: dict[str, list[Any]]) -> None:
    import pyarrow as pa

    temp_name = f"incoming_{name}"
    conn.register(temp_name, pa.table(columns))
    try:
        conn.execute(f"CREATE TABLE {name} AS SELECT * FROM {temp_name}")
    finally:
        conn.unregister(temp_name)


def create_partial_ukraine_suspended_ids(conn: Any, data_root: Path) -> int:
    import numpy as np

    path = data_root / "social_llm_data/ukr_rus_suspended/user_ids.npy"
    values = np.load(path, allow_pickle=False)
    ids = sorted(set(str(int(value)) for value in values.tolist()))
    register_arrow_table(conn, "ids_ukraine_suspended", {"id": ids})
    return len(ids)


def load_social_bios(conn: Any, data_root: Path, name: str, embedding_rel: str, csv_rel: str) -> dict[str, Any]:
    import pandas as pd
    import torch

    from scripts.bio_embeddings.preprocessing import bio_hash, normalize_bio_text

    embedding_path = data_root / embedding_rel
    obj = torch.load(embedding_path, map_location="cpu", weights_only=False)
    hashes = [str(value) if value else "" for value in obj["bio_hashes"]]
    user_ids = [str(value) for value in obj["user_ids"]]
    if len(hashes) != GRAPH_NODES[name] or len(user_ids) != GRAPH_NODES[name]:
        raise AssertionError(f"{name}: embedding rows do not match registered graph nodes")

    csv_path = data_root / csv_rel
    profiles = pd.read_csv(csv_path, usecols=["profile"])["profile"].fillna("")
    lengths: dict[str, int] = {}
    for value in profiles:
        normalized = normalize_bio_text(value)
        if normalized:
            lengths.setdefault(bio_hash(normalized), len(normalized))
    bio_len = [lengths.get(value) if value else None for value in hashes]
    key = TABLE_KEY[name]
    register_arrow_table(
        conn,
        f"bios_{key}",
        {"id": user_ids, "bio_hash": hashes, "bio_len": bio_len},
    )
    conn.execute(f"DELETE FROM bios_{key} WHERE bio_hash = ''")
    return {
        "embedding_sha256": sha256_file(embedding_path),
        "profile_source_sha256": sha256_file(csv_path),
        "bio_length_resolved": sum(value is not None for value in bio_len),
    }


def create_twibot_bios(conn: Any, data_root: Path) -> dict[str, Any]:
    edges = data_root / "twibot20/graph_build/retweet_edges.parquet"
    labels = data_root / "twibot20/raw/Twibot-20/label.csv"
    bio_root = data_root / "twibot20/bio_embeddings/gte-multilingual-base/version=v001"
    conn.execute(
        """
        CREATE TABLE ids_twibot20_internal AS
        SELECT DISTINCT CAST(userid AS VARCHAR) AS id FROM read_parquet(?)
        UNION SELECT DISTINCT CAST(rt_userid AS VARCHAR) FROM read_parquet(?)
        UNION SELECT DISTINCT CAST(id AS VARCHAR) FROM read_csv_auto(?, header=true)
        """,
        [str(edges), str(edges), str(labels)],
    )
    conn.execute(
        """
        CREATE TABLE bios_twibot20 AS
        SELECT CAST(o.userid AS VARCHAR) AS id,
               CAST(o.bio_hash AS VARCHAR) AS bio_hash,
               length(t.normalized_bio_text)::BIGINT AS bio_len
        FROM read_parquet(?) AS o
        INNER JOIN ids_twibot20_internal AS i ON CAST(o.userid AS VARCHAR) = i.id
        LEFT JOIN read_parquet(?) AS t USING (bio_hash)
        WHERE o.bio_hash IS NOT NULL AND trim(CAST(o.bio_hash AS VARCHAR)) <> ''
        QUALIFY row_number() OVER (
            PARTITION BY CAST(o.userid AS VARCHAR)
            ORDER BY COALESCE(o.last_seen_at, o.first_seen_at) DESC NULLS LAST,
                     o.bio_hash DESC
        ) = 1
        """,
        [str(bio_root / "user_bio_observations.parquet"), str(bio_root / "bio_texts.parquet")],
    )
    return {"edges_sha256": sha256_file(edges), "labels_sha256": sha256_file(labels)}


def create_hongkong_tables(conn: Any, data_root: Path) -> dict[str, Any]:
    import pyarrow.parquet as pq

    from scripts.bio_embeddings.preprocessing import bio_hash, normalize_bio_text

    users_path = data_root / "cp_hk_twitter/parquet/user_bios.parquet"
    table = pq.read_table(users_path, columns=["user_id", "profile"])
    user_ids = [str(value) for value in table.column("user_id").to_pylist()]
    normalized = [normalize_bio_text(value) for value in table.column("profile").to_pylist()]
    hashes = [bio_hash(value) if value else "" for value in normalized]
    register_arrow_table(conn, "ids_hongkong_internal", {"id": user_ids})
    register_arrow_table(
        conn,
        "bios_hongkong",
        {
            "id": user_ids,
            "bio_hash": hashes,
            "bio_len": [len(value) if value else None for value in normalized],
        },
    )
    conn.execute("DELETE FROM bios_hongkong WHERE bio_hash = ''")
    events = sorted((data_root / "cp_hk_twitter/parquet/retweet_events").glob("part-*.parquet"))
    window = conn.execute(
        "SELECT min(timestamp_ms), max(timestamp_ms) FROM read_parquet(?)", [[str(path) for path in events]]
    ).fetchone()
    return {
        "user_bios_sha256": sha256_file(users_path),
        "event_files": len(events),
        "event_manifest_sha256": source_manifest_digest(events),
        "window_start_ms": window[0],
        "window_end_ms": window[1],
    }


def table_count(conn: Any, table: str) -> int:
    return int(conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0])


def build_identity_rows(conn: Any, id_counts: dict[str, int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, dataset_a in enumerate(ALL_DATASETS):
        for dataset_b in ALL_DATASETS[index + 1 :]:
            scope_a, scope_b = EXACT_ID_SCOPE[dataset_a], EXACT_ID_SCOPE[dataset_b]
            status, note = pair_status(scope_a, scope_b)
            row: dict[str, Any] = {
                "dataset_a": dataset_a,
                "dataset_b": dataset_b,
                "status": status,
                "scope_a": scope_a,
                "scope_b": scope_b,
                "ids_a": id_counts.get(dataset_a, ""),
                "ids_b": id_counts.get(dataset_b, ""),
                "intersection": "",
                "fraction_a": "",
                "fraction_b": "",
                "fraction_smaller": "",
                "jaccard": "",
                "note": note,
            }
            if status in {"exact", "partial_exact"}:
                key_a, key_b = TABLE_KEY[dataset_a], TABLE_KEY[dataset_b]
                intersection = int(
                    conn.execute(
                        f"SELECT count(*) FROM ids_{key_a} a INNER JOIN ids_{key_b} b USING (id)"
                    ).fetchone()[0]
                )
                metrics = overlap_metrics(id_counts[dataset_a], id_counts[dataset_b], intersection)
                row.update({"intersection": intersection, **metrics})
            rows.append(row)
    return rows


def build_bio_rows(conn: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, dataset_a in enumerate(TWITTER_DATASETS):
        for dataset_b in TWITTER_DATASETS[index + 1 :]:
            key_a, key_b = TABLE_KEY[dataset_a], TABLE_KEY[dataset_b]
            result = conn.execute(
                f"""
                WITH a AS (
                    SELECT bio_hash, count(*)::BIGINT AS nodes_a, max(bio_len)::BIGINT AS bio_len
                    FROM bios_{key_a} GROUP BY bio_hash
                ), b AS (
                    SELECT bio_hash, count(*)::BIGINT AS nodes_b, max(bio_len)::BIGINT AS bio_len
                    FROM bios_{key_b} GROUP BY bio_hash
                )
                SELECT
                    count(*)::BIGINT AS shared_hashes,
                    COALESCE(sum(nodes_a), 0)::BIGINT AS nodes_a_with_shared_hash,
                    COALESCE(sum(nodes_b), 0)::BIGINT AS nodes_b_with_shared_hash,
                    count(*) FILTER (
                        WHERE nodes_a = 1 AND nodes_b = 1
                          AND COALESCE(a.bio_len, b.bio_len, 0) >= 20
                    )::BIGINT AS unique_long_shared_hashes
                FROM a INNER JOIN b USING (bio_hash)
                """
            ).fetchone()
            rows.append(
                {
                    "dataset_a": dataset_a,
                    "dataset_b": dataset_b,
                    "status": "proxy_only",
                    "shared_nonempty_bio_hashes": int(result[0]),
                    "nodes_a_with_shared_hash": int(result[1]),
                    "nodes_b_with_shared_hash": int(result[2]),
                    "unique_long_shared_hashes": int(result[3]),
                    "note": "exact normalized biography equality; unique-long remains a proxy, not identity proof",
                }
            )
    return rows


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    import duckdb

    database_path = output_dir / "identity_overlap.duckdb"
    conn = duckdb.connect(str(database_path))
    configure_duckdb(conn, output_dir, args.memory_limit, args.threads)
    provenance: dict[str, Any] = {}

    try:
        for name, spec in RAW_GRAPH_SPECS.items():
            print(f"[audit] building graph-aligned IDs and bios: {name}", flush=True)
            provenance[name] = create_raw_graph_tables(conn, data_root, name, spec)

        print("[audit] loading partial Ukraine-Suspended stable IDs", flush=True)
        partial_count = create_partial_ukraine_suspended_ids(conn, data_root)

        for name, embedding_rel in SOCIAL_EMBEDDINGS.items():
            print(f"[audit] loading graph-aligned biography hashes: {name}", flush=True)
            provenance.setdefault(name, {}).update(
                load_social_bios(conn, data_root, name, embedding_rel, SOCIAL_CSVS[name])
            )

        print("[audit] loading TwiBot-20 graph-aligned biography hashes", flush=True)
        provenance["twibot20"] = create_twibot_bios(conn, data_root)
        print("[audit] loading Hong Kong IDs and biographies", flush=True)
        provenance["hongkong"] = create_hongkong_tables(conn, data_root)

        id_counts = {
            name: table_count(conn, f"ids_{TABLE_KEY[name]}")
            for name in ("ukraine", "covid", "midterm", "ukraine-suspended")
        }
        if id_counts["ukraine-suspended"] != partial_count:
            raise AssertionError("Ukraine-Suspended ID table changed during construction")

        bio_counts = {
            name: table_count(conn, f"bios_{TABLE_KEY[name]}") for name in TWITTER_DATASETS
        }
        for name in ("ukraine", "covid", "midterm"):
            if id_counts[name] != GRAPH_NODES[name]:
                raise AssertionError(
                    f"{name}: reconstructed ID count {id_counts[name]} != graph nodes {GRAPH_NODES[name]}"
                )

        inventory_rows = []
        for name in ALL_DATASETS:
            inventory_rows.append(
                {
                    "dataset": name,
                    "platform": "facebook" if name == "facebook-page-reference" else "twitter",
                    "graph_nodes": GRAPH_NODES[name],
                    "exact_id_scope": EXACT_ID_SCOPE[name],
                    "comparable_ids": id_counts.get(name, ""),
                    "bio_hash_nodes": bio_counts.get(name, ""),
                    "bio_hash_fraction": (
                        bio_counts[name] / GRAPH_NODES[name] if name in bio_counts else ""
                    ),
                }
            )

        identity_rows = build_identity_rows(conn, id_counts)
        bio_rows = build_bio_rows(conn)
    finally:
        conn.close()

    write_csv(
        output_dir / "dataset_inventory.csv",
        inventory_rows,
        [
            "dataset",
            "platform",
            "graph_nodes",
            "exact_id_scope",
            "comparable_ids",
            "bio_hash_nodes",
            "bio_hash_fraction",
        ],
    )
    write_csv(
        output_dir / "pairwise_identity_overlap.csv",
        identity_rows,
        [
            "dataset_a",
            "dataset_b",
            "status",
            "scope_a",
            "scope_b",
            "ids_a",
            "ids_b",
            "intersection",
            "fraction_a",
            "fraction_b",
            "fraction_smaller",
            "jaccard",
            "note",
        ],
    )
    write_csv(
        output_dir / "pairwise_biography_overlap.csv",
        bio_rows,
        [
            "dataset_a",
            "dataset_b",
            "status",
            "shared_nonempty_bio_hashes",
            "nodes_a_with_shared_hash",
            "nodes_b_with_shared_hash",
            "unique_long_shared_hashes",
            "note",
        ],
    )

    measurable = [row for row in identity_rows if row["status"] in {"exact", "partial_exact"}]
    summary = {
        "protocol": {
            "identity": "exact comparable platform-global Twitter user ID",
            "biography": "SHA-256 equality after bio-text-v001 normalization",
            "strict_biography_proxy": "shared hash, normalized length >=20, unique within each graph",
            "privacy": "aggregate counts only; no IDs, handles, or biography text emitted",
        },
        "datasets": ALL_DATASETS,
        "graph_nodes": GRAPH_NODES,
        "exact_id_scope": EXACT_ID_SCOPE,
        "provenance": provenance,
        "headline": {
            "measurable_id_pairs": len(measurable),
            "max_exact_id_intersection": max(int(row["intersection"]) for row in measurable),
            "max_exact_id_fraction_smaller": max(float(row["fraction_smaller"]) for row in measurable),
            "bio_proxy_pairs": len(bio_rows),
            "max_unique_long_shared_bios": max(row["unique_long_shared_hashes"] for row in bio_rows),
        },
    }
    summary["output_sha256"] = {
        filename: sha256_file(output_dir / filename)
        for filename in (
            "dataset_inventory.csv",
            "pairwise_identity_overlap.csv",
            "pairwise_biography_overlap.csv",
        )
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary["headline"], indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
