from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METRICS = ("roc_auc_ovr_macro", "f1_macro", "accuracy", "balanced_accuracy")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-root", required=True)
    parser.add_argument("--finetune-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = []
    probe_files = sorted(Path(args.probe_root).glob("*.json"))
    finetune_files = sorted(Path(args.finetune_root).glob("*/summary.json"))
    if len(probe_files) != 6 or len(finetune_files) != 6:
        raise ValueError(f"expected 6 probe and 6 fine-tune results, found {len(probe_files)} and {len(finetune_files)}")
    split_hashes = set()
    for path in probe_files:
        result = json.loads(path.read_text())
        split_hashes.add(result["target_split_hash"])
        rows.append(
            {
                "initialization": path.stem.removeprefix("probe_"),
                "evaluation": "frozen_linear_probe",
                "sources": ",".join(result["sources"]),
                "selected_step": result["checkpoint_step"],
                "selected_epoch": "",
                "train_nodes": result["train_nodes"],
                "validation_nodes": result["validation_nodes"],
                "test_nodes": result["test_nodes"],
                **{metric: result[metric] for metric in METRICS},
            }
        )
    for path in finetune_files:
        result = json.loads(path.read_text())
        split_hashes.add(result["target_split_hash"])
        if int(result["test_evaluations"]) != 1:
            raise ValueError(f"{path}: expected exactly one test evaluation")
        rows.append(
            {
                "initialization": path.parent.name.removeprefix("ft_").removesuffix("_s0"),
                "evaluation": "supervised_finetune",
                "sources": ",".join(result["initial_sources"]),
                "selected_step": result["pretrain_step"],
                "selected_epoch": result["best_epoch"],
                "train_nodes": result["train_nodes"],
                "validation_nodes": result["validation_nodes"],
                "test_nodes": result["test_nodes"],
                **{metric: result[metric] for metric in METRICS},
            }
        )
    if len(split_hashes) != 1:
        raise ValueError(f"results used different target splits: {split_hashes}")
    node_counts = {(row["train_nodes"], row["validation_nodes"], row["test_nodes"]) for row in rows}
    if len(node_counts) != 1:
        raise ValueError(f"results used different target counts: {node_counts}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(f"wrote {output}: {len(rows)} rows, split={next(iter(split_hashes))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
