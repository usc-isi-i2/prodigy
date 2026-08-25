#!/usr/bin/env python3
"""Build a readable Markdown inventory of all exported W&B runs."""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
INPUT = ROOT / "wandb_exports" / "graph_clip_run_metadata.csv"
OUTPUT = Path(__file__).resolve().parent / "WANDB_EXPERIMENT_INVENTORY.md"


def family(name: str) -> str:
    value = name or "(unnamed)"
    rules = [
        (r"^eval_", "evaluation runs"),
        (r"^finalcore", "final-core experiments"),
        (r"^nm_ladder", "neighbor-matching ladders"),
        (r"^radiusfc", "radius/fixed-context experiments"),
        (r"^TWITTER_PT_PRODIGY", "PRODIGY Twitter pretraining"),
        (r"^TWITTER_PT", "Twitter pretraining"),
        (r"^samgpt", "SAMGPT / GraphCL"),
        (r"^cov_", "multitask SSL corpora"),
        (r"^muc", "MUC / regression probes"),
        (r"^sat", "saturation studies"),
        (r"^train", "training runs"),
    ]
    for pattern, label in rules:
        if re.search(pattern, value, re.IGNORECASE):
            return label
    return "other / uncategorized"


def main():
    with INPUT.open(newline="", encoding="utf-8") as handle:
        runs = list(csv.DictReader(handle))
    runs.sort(key=lambda row: (row.get("created_at", ""), row.get("run_id", "")), reverse=True)

    month_counts = Counter(row.get("created_at", "")[:7] or "unknown" for row in runs)
    status_counts = Counter(row.get("state", "unknown") or "unknown" for row in runs)
    family_counts = Counter(family(row.get("run_name", "")) for row in runs)

    lines = [
        "# W&B experiment inventory",
        "",
        "This document inventories every run in the exported `eibl-usc/graph-clip` W&B project.",
        "It is a run archive, not a metric-clean evaluation table. The canonical metric ledger is maintained separately.",
        "",
        f"- Total runs: **{len(runs):,}**",
        f"- Earliest run: **{min((r.get('created_at','') for r in runs), default='unknown')}**",
        f"- Latest run: **{max((r.get('created_at','') for r in runs), default='unknown')}**",
        "- Source: `wandb_exports/graph_clip_run_metadata.csv`",
        "",
        "## Runs by month",
        "",
        "| Month (UTC) | Runs |",
        "|---|---:|",
    ]
    lines.extend(f"| {month} | {count:,} |" for month, count in sorted(month_counts.items()))
    lines.extend(["", "## Runs by status", "", "| Status | Runs |", "|---|---:|"])
    lines.extend(f"| {status} | {count:,} |" for status, count in sorted(status_counts.items()))
    lines.extend(["", "## Runs by experiment family", "", "| Family | Runs |", "|---|---:|"])
    lines.extend(f"| {label} | {count:,} |" for label, count in sorted(family_counts.items()))
    lines.extend(["", "## Complete run list", "", "| Created (UTC) | Status | Family | Run name | W&B ID |", "|---|---|---|---|---|"])
    for row in runs:
        name = row.get("run_name", "") or "(unnamed)"
        run_id = row.get("run_id", "")
        url = row.get("url", "") or f"https://wandb.ai/eibl-usc/graph-clip/runs/{run_id}"
        safe_name = name.replace("|", "\\|")
        lines.append(
            f"| {row.get('created_at','')} | {row.get('state','')} | {family(name)} | "
            f"[{safe_name}]({url}) | `{run_id}` |"
        )
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"runs={len(runs)} output={OUTPUT}")


if __name__ == "__main__":
    main()
