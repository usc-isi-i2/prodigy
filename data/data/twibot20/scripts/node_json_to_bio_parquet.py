#!/usr/bin/env python3
"""Convert TwiBot-20 (Format22) ``node.json`` into a bio-embedding-ready Parquet.

TwiBot-20 ships every node (users *and* tweets) as a single large JSON array in
``node.json``. User nodes carry an ``id`` prefixed with ``u`` and a profile
``description`` (the bio); tweet nodes are prefixed with ``t`` and have no bio.

The bio embedding pipeline (``scripts/bio_embeddings/embed_bios.py``) reads
Parquet via DuckDB and auto-detects a ``userid`` / ``description`` / ``created_at``
column set for the ``author`` source role. This converter streams ``node.json``
with ijson (constant memory), keeps only user nodes, and writes a single Parquet
file with exactly those columns. No tweet/retweet/quote bio columns exist in
TwiBot-20, so the pipeline naturally produces only ``author``-role observations.

The ``u`` id prefix is preserved so the output joins directly to ``label.csv``,
``split.csv`` and ``edge.csv``, which all use the same ``u...`` / ``t...`` ids.

Example
-------
    python node_json_to_bio_parquet.py \
        --input  /dataMeR1/phil/data/twibot20/raw/Twibot-20/node.json \
        --output /dataMeR1/phil/data/twibot20/parquet/users/users-000.parquet
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import ijson
import pyarrow as pa
import pyarrow.parquet as pq

# One row group per this many users. 229k users total, so this stays a single
# well-formed file with a couple of row groups.
DEFAULT_BATCH_ROWS = 50_000

SCHEMA = pa.schema(
    [
        pa.field("userid", pa.string()),
        pa.field("description", pa.string()),
        pa.field("created_at", pa.string()),
    ]
)


def _clean(value: object) -> str | None:
    """Return a stripped string, or ``None`` for missing/blank values.

    Raw bios/created_at in TwiBot-20 carry trailing spaces; the embedding
    pipeline re-normalizes bios (NFKC + trim), but trimming ``created_at`` here
    lets DuckDB's ``try_strptime`` parse the Twitter v1 timestamp cleanly.
    """
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def convert(input_path: Path, output_path: Path, batch_rows: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_nodes = n_users = n_nonempty = 0
    userids: list[str | None] = []
    descriptions: list[str | None] = []
    created_ats: list[str | None] = []
    start = time.time()

    writer = pq.ParquetWriter(output_path, SCHEMA, compression="snappy")

    def flush() -> None:
        if not userids:
            return
        batch = pa.record_batch(
            [
                pa.array(userids, type=pa.string()),
                pa.array(descriptions, type=pa.string()),
                pa.array(created_ats, type=pa.string()),
            ],
            schema=SCHEMA,
        )
        writer.write_batch(batch)
        userids.clear()
        descriptions.clear()
        created_ats.clear()

    try:
        with open(input_path, "rb") as handle:
            for obj in ijson.items(handle, "item"):
                n_nodes += 1
                node_id = str(obj.get("id", ""))
                if not node_id.startswith("u"):
                    continue  # tweet node, no bio
                n_users += 1
                desc = _clean(obj.get("description"))
                if desc is not None:
                    n_nonempty += 1
                userids.append(node_id)
                descriptions.append(desc)
                created_ats.append(_clean(obj.get("created_at")))
                if len(userids) >= batch_rows:
                    flush()
                if n_nodes % 5_000_000 == 0:
                    print(
                        f"  scanned {n_nodes:,} nodes / {n_users:,} users "
                        f"({time.time() - start:.0f}s)",
                        flush=True,
                    )
        flush()
    finally:
        writer.close()

    elapsed = time.time() - start
    print(
        f"Done in {elapsed:.0f}s: {n_nodes:,} nodes -> {n_users:,} users "
        f"({n_nonempty:,} with non-empty bio) written to {output_path}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="/dataMeR1/phil/data/twibot20/raw/Twibot-20/node.json",
        help="Path to TwiBot-20 node.json.",
    )
    parser.add_argument(
        "--output",
        default="/dataMeR1/phil/data/twibot20/parquet/users/users-000.parquet",
        help="Output Parquet path (dir is used as the pipeline input_root).",
    )
    parser.add_argument("--batch-rows", type=int, default=DEFAULT_BATCH_ROWS)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"ERROR: input not found: {input_path}", file=sys.stderr)
        return 1
    if output_path.exists() and not args.overwrite:
        print(
            f"ERROR: output exists (use --overwrite): {output_path}",
            file=sys.stderr,
        )
        return 1

    print(f"Converting {input_path} -> {output_path}", flush=True)
    convert(input_path, output_path, args.batch_rows)
    size = os.path.getsize(output_path)
    print(f"Parquet size: {size / 1e6:.1f} MB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
