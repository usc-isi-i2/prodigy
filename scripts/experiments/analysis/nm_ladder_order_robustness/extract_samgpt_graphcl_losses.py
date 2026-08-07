#!/usr/bin/env python3
"""Extract native SAMGPT GraphCL training losses for the 3-order source ladder.

The checked-in ladder tables contain downstream neighbor-matching metrics, while
the original SAMGPT run directories also contain ``row_metrics.json`` and
``pretrain_history.jsonl``.  This script reads those original artifacts and writes:

* ``graphcl_loss_summary.csv``: one row per order/rung; and
* ``graphcl_source_losses.csv``: per-source loss at the best and final epochs.

Run locally against CARC paths referenced by the canonical manifest::

    python extract_samgpt_graphcl_losses.py --ssh-host carc

Omit ``--ssh-host`` when the absolute paths in the manifest are locally mounted.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = HERE / "data" / "samgpt_9x3_carc_v100"

SUMMARY_FIELDS = [
    "order",
    "rung",
    "added",
    "source_count",
    "sources",
    "best_loss",
    "final_loss",
    "best_epoch",
    "final_epoch",
    "epochs_run",
    "training_seconds",
    "loss_aggregation",
    "history_entries",
    "row_metrics_path",
]

SOURCE_FIELDS = [
    "order",
    "rung",
    "added",
    "source_count",
    "source",
    "best_epoch_source_loss",
    "final_epoch_source_loss",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Directory containing the canonical SAMGPT ladder manifest",
    )
    parser.add_argument(
        "--ssh-host",
        help="Read run artifacts through one SSH connection (for example: carc)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: DATA_ROOT)",
    )
    return parser.parse_args()


def requested_paths(manifest: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for row_path in manifest["rows"].values():
        paths.append(row_path)
        paths.append(str(Path(row_path).with_name("pretrain_history.jsonl")))
    return paths


def read_local(paths: list[str]) -> dict[str, str]:
    contents: dict[str, str] = {}
    missing: list[str] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_file():
            contents[raw_path] = path.read_text()
        else:
            missing.append(raw_path)
    if missing:
        preview = "\n".join(f"  {path}" for path in missing[:8])
        suffix = "\n  ..." if len(missing) > 8 else ""
        raise FileNotFoundError(f"Missing {len(missing)} run artifacts:\n{preview}{suffix}")
    return contents


def read_remote(host: str, paths: list[str]) -> dict[str, str]:
    """Read every requested file in one SSH connection."""

    remote_program = """
import base64
import json
import pathlib
import sys

paths = json.loads(base64.b64decode(sys.argv[1]).decode("utf-8"))
result = {"files": {}, "missing": []}
for raw_path in paths:
    path = pathlib.Path(raw_path)
    if path.is_file():
        result["files"][raw_path] = path.read_text()
    else:
        result["missing"].append(raw_path)
print("__GRAPHCL_JSON__" + json.dumps(result))
""".strip()
    encoded_paths = base64.b64encode(json.dumps(paths).encode("utf-8")).decode("ascii")
    remote_command = (
        f"python3 -c {shlex.quote(remote_program)} {shlex.quote(encoded_paths)}"
    )
    completed = subprocess.run(
        ["ssh", "-tt", "-o", "BatchMode=yes", host, remote_command],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"SSH read from {host!r} failed: {detail}")
    response_line = next(
        (
            line.removeprefix("__GRAPHCL_JSON__")
            for line in completed.stdout.replace("\r", "").splitlines()
            if line.startswith("__GRAPHCL_JSON__")
        ),
        None,
    )
    if response_line is None:
        raise RuntimeError(
            f"SSH read from {host!r} returned no artifact payload: "
            f"{completed.stdout.strip()}"
        )
    response = json.loads(response_line)
    if response["missing"]:
        preview = "\n".join(f"  {path}" for path in response["missing"][:8])
        suffix = "\n  ..." if len(response["missing"]) > 8 else ""
        raise FileNotFoundError(
            f"Missing {len(response['missing'])} remote run artifacts:\n"
            f"{preview}{suffix}"
        )
    return response["files"]


def parse_history(text: str, path: str) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"Empty pretraining history: {path}")
    for expected_epoch, row in enumerate(rows):
        if int(row["epoch"]) != expected_epoch:
            raise ValueError(
                f"Non-contiguous epochs in {path}: expected {expected_epoch}, "
                f"found {row['epoch']}"
            )
    return rows


def extract(
    manifest: dict[str, Any], contents: dict[str, str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []

    for order in ("A", "B", "C"):
        sequence = manifest["orders"][order]
        for rung in range(1, len(sequence) + 1):
            key = f"{order}/rung_{rung:02d}"
            row_path = manifest["rows"][key]
            history_path = str(Path(row_path).with_name("pretrain_history.jsonl"))
            result = json.loads(contents[row_path])
            training = result.get("training") or result.get("pretraining")
            if not isinstance(training, dict):
                raise KeyError(f"No training record in {row_path}")

            history = parse_history(contents[history_path], history_path)
            best_epoch = int(training["best_epoch"])
            if not 0 <= best_epoch < len(history):
                raise ValueError(
                    f"best_epoch={best_epoch} is outside the history in {history_path}"
                )
            best_row = history[best_epoch]
            final_row = history[-1]
            best_loss = float(training["best_loss"])
            recorded_best_loss = float(best_row["loss"])
            if abs(best_loss - recorded_best_loss) > 1e-9:
                raise ValueError(
                    f"Best-loss mismatch in {row_path}: {best_loss} vs "
                    f"history value {recorded_best_loss}"
                )

            sources = sequence[:rung]
            summaries.append(
                {
                    "order": order,
                    "rung": rung,
                    "added": sequence[rung - 1],
                    "source_count": rung,
                    "sources": ";".join(sources),
                    "best_loss": best_loss,
                    "final_loss": float(final_row["loss"]),
                    "best_epoch": best_epoch,
                    "final_epoch": int(final_row["epoch"]),
                    "epochs_run": int(training["epochs_run"]),
                    "training_seconds": float(training["seconds"]),
                    "loss_aggregation": training.get("loss_aggregation", ""),
                    "history_entries": len(history),
                    "row_metrics_path": row_path,
                }
            )

            best_source_losses = best_row.get("source_losses", {})
            final_source_losses = final_row.get("source_losses", {})
            if set(best_source_losses) != set(sources):
                raise ValueError(
                    f"Source-loss keys in {history_path} do not match manifest: "
                    f"{sorted(best_source_losses)} vs {sorted(sources)}"
                )
            for source in sources:
                source_rows.append(
                    {
                        "order": order,
                        "rung": rung,
                        "added": sequence[rung - 1],
                        "source_count": rung,
                        "source": source,
                        "best_epoch_source_loss": float(best_source_losses[source]),
                        "final_epoch_source_loss": float(final_source_losses[source]),
                    }
                )

    return summaries, source_rows


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    manifest_path = data_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    paths = requested_paths(manifest)
    contents = (
        read_remote(args.ssh_host, paths) if args.ssh_host else read_local(paths)
    )
    summaries, source_rows = extract(manifest, contents)
    if len(summaries) != 27:
        raise ValueError(f"Expected 27 rung summaries, found {len(summaries)}")

    output_dir = (args.output_dir or data_root).resolve()
    write_csv(output_dir / "graphcl_loss_summary.csv", SUMMARY_FIELDS, summaries)
    write_csv(output_dir / "graphcl_source_losses.csv", SOURCE_FIELDS, source_rows)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
