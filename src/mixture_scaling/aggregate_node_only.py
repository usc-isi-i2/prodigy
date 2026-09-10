from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .evaluate_lattice import LP_TARGETS
from .lattice import SOURCE_ORDER
from .node_only_transfer import singleton_rows


def write_tsv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def collect(state_root: Path, results_root: Path, objective: str) -> list[dict]:
    targets = SOURCE_ORDER if objective == "fp" else LP_TARGETS
    rows, missing = [], []
    for run_id, (source,) in singleton_rows():
        summary_path = state_root / objective / run_id / "summary.json"
        best_path = state_root / objective / run_id / "best.pt"
        if not summary_path.is_file() or not best_path.is_file():
            missing.append(f"training:{run_id}")
            continue
        summary = json.loads(summary_path.read_text())
        if summary.get("status") != "complete" or summary.get("sources") != [source]:
            raise ValueError(f"invalid training summary: {summary_path}")
        for target in targets:
            path = results_root / objective / f"{run_id}__to__{target}.json"
            if not path.is_file():
                missing.append(f"evaluation:{run_id}->{target}")
                continue
            payload = json.loads(path.read_text())
            if payload.get("status") != "complete":
                missing.append(f"evaluation:{run_id}->{target}")
                continue
            row = {
                "objective": objective, "architecture": "node_mlp", "source": source,
                "target": target, "checkpoint_step": payload["checkpoint_step"],
            }
            if objective == "fp":
                row.update({
                    "metric": "scaled_cosine_error", "value": payload["scaled_cosine_error_mean"],
                    "replicate_std": payload["scaled_cosine_error_std"],
                })
            else:
                gates = payload["gates"]
                if gates["holdout_leakage_edges"] != 0 or gates["endpoint_sensitivity"] <= 0:
                    raise ValueError(f"failed LP gate: {path}")
                row.update({"metric": "roc_auc", "value": payload["report"]["auc"], "replicate_std": ""})
            rows.append(row)
    if missing:
        raise FileNotFoundError(f"missing/incomplete artifacts ({len(missing)}): {missing[:20]}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--objective", choices=("lp", "fp"), required=True)
    parser.add_argument("--state-root", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    rows = collect(Path(args.state_root), Path(args.results_root), args.objective)
    output = Path(args.output_root)
    write_tsv(output / f"node_mlp_{args.objective}_transfer.tsv", rows)
    receipt = {
        "status": "complete", "objective": args.objective, "architecture": "node_mlp",
        "training_models": len(singleton_rows()), "evaluation_cells": len(rows),
    }
    (output / f"node_mlp_{args.objective}_COMPLETE.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(json.dumps(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
