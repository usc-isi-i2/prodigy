from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .evaluate_lattice import CLS_TARGETS, LP_TARGETS
from .lattice import lattice_rows


def verify_training(state_root: Path, objective: str) -> list[dict]:
    rows = []
    missing = []
    for run_id, sources in lattice_rows():
        run_dir = state_root / objective / run_id
        summary_path = run_dir / "summary.json"
        best = run_dir / "best.pt"
        if not summary_path.is_file() or not best.is_file():
            missing.append(run_id)
            continue
        summary = json.loads(summary_path.read_text())
        terminal = run_dir / "checkpoints" / f"step_{summary['final_step']}.pt"
        if summary.get("status") != "complete" or not terminal.is_file():
            missing.append(run_id)
            continue
        if tuple(summary["sources"]) != sources:
            raise ValueError(f"source mismatch for {run_id}")
        if not list((run_dir / "wandb").glob("offline-run-*")):
            raise ValueError(f"offline W&B run missing for {run_id}")
        rows.append({
            "run_id": run_id, "sources": ",".join(sources), "mixture_size": len(sources),
            "best_step": summary["best_step"], "final_step": summary["final_step"],
            "best_validation_loss": summary["best_validation_loss"],
            "elapsed_seconds": summary["elapsed_seconds"],
        })
    if missing:
        raise FileNotFoundError(f"missing/incomplete training runs ({len(missing)}): {missing}")
    return rows


def collect_evaluation(results_root: Path, objective: str, task: str) -> list[dict]:
    targets = CLS_TARGETS if task == "cls" else LP_TARGETS
    rows = []
    missing = []
    for run_id, sources in lattice_rows():
        for target in targets:
            path = results_root / objective / task / f"{run_id}__to__{target}.json"
            if not path.is_file():
                missing.append(f"{run_id}->{target}")
                continue
            payload = json.loads(path.read_text())
            if payload.get("status") != "complete":
                missing.append(f"{run_id}->{target}")
                continue
            if task == "cls":
                auc = payload["roc_auc_ovr_macro"]
            else:
                auc = payload["report"]["auc"]
                if payload["gates"]["holdout_leakage_edges"] != 0:
                    raise ValueError(f"holdout leakage in {path}")
                if payload["gates"]["endpoint_sensitivity"] <= 0:
                    raise ValueError(f"endpoint-insensitive scorer in {path}")
            rows.append({
                "objective": objective, "task": task, "run_id": run_id,
                "sources": ",".join(sources), "mixture_size": len(sources),
                "target": target, "auc": auc, "checkpoint_step": payload["checkpoint_step"],
            })
    if missing:
        raise FileNotFoundError(f"missing/incomplete eval cells ({len(missing)}): {missing[:20]}")
    return rows


def write_tsv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--objective", choices=("lp", "graphmae"), required=True)
    parser.add_argument("--state-root", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    output = Path(args.output_root)
    training = verify_training(Path(args.state_root), args.objective)
    cls = collect_evaluation(Path(args.results_root), args.objective, "cls")
    lp = collect_evaluation(Path(args.results_root), args.objective, "lp")
    write_tsv(output / f"{args.objective}_training.tsv", training)
    write_tsv(output / f"{args.objective}_classification.tsv", cls)
    write_tsv(output / f"{args.objective}_static_link_prediction.tsv", lp)
    receipt = {
        "objective": args.objective, "training_models": len(training),
        "classification_cells": len(cls), "static_link_prediction_cells": len(lp),
        "status": "complete",
    }
    (output / f"{args.objective}_COMPLETE.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
