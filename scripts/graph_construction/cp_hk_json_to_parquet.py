"""Extract CP-HK Twitter retweet events and user bios from COSINE JSONL gzip.

Input files are COSINE Twitter JSON lines with hashed user identifiers and
masked bio/text fields. The output is intentionally narrow:

* ``retweet_events/part-*.parquet`` contains one row per retweet event.
* ``user_bios.parquet`` contains one normalized bio row per observed user.
* ``manifest.json`` records counts, inputs, and command provenance.

The downstream graph builder consumes these files directly.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import shlex
import sys
import time
from typing import Any

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.bio_embeddings.preprocessing import normalize_bio_text


EVENT_COLUMNS = [
    "source_user_id",
    "target_user_id",
    "tweet_id",
    "retweeted_tweet_id",
    "timestamp_ms",
    "created_at",
    "lang",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, help="Input .json.gz path. Repeatable.")
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    parser.add_argument("--flush-rows", type=int, default=250_000)
    parser.add_argument("--max-records", type=int, default=0, help="Debug cap across all inputs; 0 means no cap.")
    return parser.parse_args()


def _get_user_id(user: dict[str, Any]) -> str:
    return str(user.get("id_h") or user.get("id_str_h") or user.get("user_id_h") or "").strip()


def _get_description(user: dict[str, Any]) -> str:
    return str(user.get("description_m") or user.get("description") or user.get("description_h") or "").strip()


def _observe_user(user_bios: dict[str, dict[str, Any]], user: dict[str, Any], timestamp_ms: int | None) -> None:
    user_id = _get_user_id(user)
    if not user_id:
        return
    raw_bio = _get_description(user)
    normalized = normalize_bio_text(raw_bio)
    if not normalized and user_id in user_bios:
        return

    previous = user_bios.get(user_id)
    if previous is None:
        user_bios[user_id] = {
            "user_id": user_id,
            "profile": normalized,
            "raw_profile": raw_bio,
            "last_timestamp_ms": timestamp_ms if timestamp_ms is not None else -1,
            "observations": 1,
        }
        return

    previous["observations"] += 1
    previous_ts = int(previous.get("last_timestamp_ms", -1))
    curr_ts = timestamp_ms if timestamp_ms is not None else -1
    # Prefer a non-empty newer bio. If timestamps tie, keep the longer text.
    if normalized and (curr_ts > previous_ts or (curr_ts == previous_ts and len(normalized) > len(previous["profile"]))):
        previous["profile"] = normalized
        previous["raw_profile"] = raw_bio
        previous["last_timestamp_ms"] = curr_ts


def _parse_timestamp_ms(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _write_events(rows: list[dict[str, Any]], out_dir: Path, part_idx: int) -> None:
    if not rows:
        return
    part_path = out_dir / "retweet_events" / f"part-{part_idx:05d}.parquet"
    pd.DataFrame(rows, columns=EVENT_COLUMNS).to_parquet(part_path, index=False)


def main() -> None:
    args = parse_args()
    started = time.time()
    out_dir = Path(args.out_dir)
    event_dir = out_dir / "retweet_events"
    event_dir.mkdir(parents=True, exist_ok=True)
    for stale_part in event_dir.glob("part-*.parquet"):
        stale_part.unlink()

    command = " ".join(shlex.quote(x) for x in [sys.executable, *sys.argv])
    event_rows: list[dict[str, Any]] = []
    user_bios: dict[str, dict[str, Any]] = {}
    part_idx = 0
    total_rows = 0
    retweet_rows = 0
    skipped_rows = 0

    for input_path in args.input:
        path = Path(input_path)
        print(f"[cp_hk_json_to_parquet] reading {path}", flush=True)
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if args.max_records and total_rows >= args.max_records:
                    break
                total_rows += 1
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    skipped_rows += 1
                    continue

                timestamp_ms = _parse_timestamp_ms(obj.get("timestamp_ms"))
                source_user = obj.get("user") or {}
                _observe_user(user_bios, source_user, timestamp_ms)

                retweeted = obj.get("retweeted_status")
                if not isinstance(retweeted, dict):
                    continue
                target_user = retweeted.get("user") or {}
                _observe_user(user_bios, target_user, timestamp_ms)

                source_id = _get_user_id(source_user)
                target_id = _get_user_id(target_user)
                if not source_id or not target_id or source_id == target_id:
                    skipped_rows += 1
                    continue

                event_rows.append(
                    {
                        "source_user_id": source_id,
                        "target_user_id": target_id,
                        "tweet_id": str(obj.get("id_h") or obj.get("id_str_h") or ""),
                        "retweeted_tweet_id": str(retweeted.get("id_h") or retweeted.get("id_str_h") or ""),
                        "timestamp_ms": timestamp_ms,
                        "created_at": str(obj.get("created_at") or ""),
                        "lang": str(obj.get("lang") or ""),
                    }
                )
                retweet_rows += 1

                if len(event_rows) >= args.flush_rows:
                    _write_events(event_rows, out_dir, part_idx)
                    print(
                        f"[cp_hk_json_to_parquet] wrote part={part_idx} "
                        f"events={retweet_rows:,} rows_seen={total_rows:,} users={len(user_bios):,}",
                        flush=True,
                    )
                    event_rows.clear()
                    part_idx += 1
            if args.max_records and total_rows >= args.max_records:
                break

    _write_events(event_rows, out_dir, part_idx)

    users_df = pd.DataFrame(sorted(user_bios.values(), key=lambda row: row["user_id"]))
    users_df.insert(0, "node_id", range(len(users_df)))
    users_df.to_parquet(out_dir / "user_bios.parquet", index=False)

    manifest = {
        "inputs": args.input,
        "out_dir": str(out_dir),
        "total_rows": total_rows,
        "retweet_rows": retweet_rows,
        "skipped_rows": skipped_rows,
        "users": int(len(users_df)),
        "users_with_bio": int((users_df["profile"].fillna("").str.len() > 0).sum()),
        "event_parts": len(list(event_dir.glob("part-*.parquet"))),
        "command": command,
        "wall_min": round((time.time() - started) / 60, 2),
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
