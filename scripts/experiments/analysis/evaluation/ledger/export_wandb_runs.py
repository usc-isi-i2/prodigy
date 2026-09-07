#!/usr/bin/env python3
"""Export W&B run metadata/config/summary data for later ledger ingestion."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import wandb


def safe_json(value):
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except TypeError:
        return json.dumps(str(value), ensure_ascii=False)


def safe_attr(run, name):
    value = getattr(run, name, "")
    return "" if value is None else value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("project", nargs="?", default="eibl-usc/graph-clip")
    parser.add_argument("--output", type=Path, default=Path("wandb_exports/graph_clip_runs.csv"))
    parser.add_argument("--limit", type=int, default=0, help="Optional maximum number of runs; 0 means all")
    parser.add_argument("--metadata-only", action="store_true", help="Avoid per-run summary requests")
    args = parser.parse_args()

    api = wandb.Api()
    fields = [
        "project", "run_id", "run_name", "display_name", "state", "created_at",
        "updated_at", "heartbeat_at", "group", "job_type", "tags_json", "url",
        "path", "summary_json", "config_json",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        handle.flush()
        runs = api.runs(args.project, per_page=100)
        for run in runs:
            if args.limit and count >= args.limit:
                break
            attrs = getattr(run, "_attrs", {}) or {}
            if args.metadata_only:
                summary = attrs.get("summaryMetrics", {}) or {}
                config = attrs.get("config", {}) or attrs.get("rawconfig", {}) or {}
            else:
                summary = getattr(run.summary, "_json_dict", {}) or {}
                config = {
                    key: value
                    for key, value in (run.config or {}).items()
                    if not str(key).startswith("_")
                }
            writer.writerow(
                {
                    "project": args.project,
                    "run_id": safe_attr(run, "id"),
                    "run_name": safe_attr(run, "name"),
                    "display_name": safe_attr(run, "display_name"),
                    "state": safe_attr(run, "state"),
                    "created_at": safe_attr(run, "created_at"),
                    "updated_at": safe_attr(run, "updated_at"),
                    "heartbeat_at": safe_attr(run, "heartbeat_at"),
                    "group": safe_attr(run, "group"),
                    "job_type": safe_attr(run, "job_type"),
                    "tags_json": safe_json(list(safe_attr(run, "tags") or [])),
                    "url": safe_attr(run, "url"),
                    "path": "/".join(safe_attr(run, "path") or []),
                    "summary_json": safe_json(summary),
                    "config_json": safe_json(config),
                }
            )
            count += 1
            if count % 100 == 0:
                handle.flush()
                print(f"exported={count}", flush=True)
    print(f"project={args.project} runs={count} output={args.output}")


if __name__ == "__main__":
    main()
