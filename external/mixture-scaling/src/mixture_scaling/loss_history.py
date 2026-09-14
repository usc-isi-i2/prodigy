from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def run_kind(run_id: str) -> str:
    if run_id.startswith("specialist_"):
        return "specialist"
    if run_id.startswith("loo_"):
        return "leave_one_out"
    if run_id.startswith("ladder_k"):
        return "ladder"
    raise ValueError(f"cannot infer run kind from {run_id}")


def collect(state_roots: list[tuple[str, Path]]) -> tuple[list[dict], list[dict]]:
    histories: list[dict] = []
    per_source: list[dict] = []
    for experiment, state_root in state_roots:
        for run_dir in sorted(path for path in state_root.iterdir() if path.is_dir()):
            history_path = run_dir / "validation.jsonl"
            metadata_path = run_dir / "metadata.json"
            summary_path = run_dir / "summary.json"
            if not (history_path.is_file() and metadata_path.is_file() and summary_path.is_file()):
                raise ValueError(f"incomplete run directory: {run_dir}")
            metadata = json.loads(metadata_path.read_text())
            summary = json.loads(summary_path.read_text())
            if summary.get("status") != "complete":
                raise ValueError(f"run is not complete: {run_dir}")
            sources = metadata["sources"]
            common = {
                "experiment": experiment,
                "run_id": run_dir.name,
                "kind": run_kind(run_dir.name),
                "seed": int(metadata["seed"]),
                "mixture_size": len(sources),
                "sources": ",".join(sources),
                "best_step": int(summary["best_step"]),
                "final_step": int(summary["final_step"]),
            }
            entries = [json.loads(line) for line in history_path.read_text().splitlines() if line.strip()]
            if not entries or int(entries[-1]["step"]) != int(summary["final_step"]):
                raise ValueError(f"history/final-step mismatch: {run_dir}")
            for entry in entries:
                row = {
                    **common,
                    "step": int(entry["step"]),
                    "train_loss": float(entry["train_loss"]),
                    "validation_loss": float(entry["validation_loss"]),
                    "elapsed_seconds": float(entry["elapsed_seconds"]),
                }
                histories.append(row)
                for source, loss in entry["per_source_validation_loss"].items():
                    per_source.append({**common, "step": int(entry["step"]), "validation_source": source, "validation_loss": float(loss)})
    return histories, per_source


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-root", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    state_roots = []
    for value in args.state_root:
        label, separator, path = value.partition("=")
        if not separator:
            raise ValueError(f"state root must be LABEL=PATH: {value}")
        state_roots.append((label, Path(path)))
    histories, per_source = collect(state_roots)
    output_root = Path(args.output_root)
    write_csv(output_root / "loss_history.csv", histories)
    write_csv(output_root / "per_source_validation_loss.csv", per_source)
    print(f"wrote {len(histories)} history rows and {len(per_source)} per-source rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
