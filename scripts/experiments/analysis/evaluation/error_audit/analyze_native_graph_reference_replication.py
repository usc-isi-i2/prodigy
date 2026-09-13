"""Combine exact class-reference receipts for the canonical HK/Ukraine 2x2 panel."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", action="append", nargs=3, metavar=("TARGET", "MODEL", "RECEIPT"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cells = {}
    hashes = {}
    for target, model, path_text in args.cell:
        path = Path(path_text)
        receipt = json.loads(path.read_text())
        assert receipt["complete"] and receipt["target"] == target and receipt["model"] == model
        assert receipt["all"]["rows"] == 61_440 and receipt["unchanged_model"]
        cells[f"{model}_to_{target}"] = receipt["all"]
        hashes[f"{model}_to_{target}"] = {
            "model_state_sha256": receipt["model_state_sha256"],
            "rows_sha256": receipt["rows_sha256"],
            "max_additive_error": receipt["max_additive_error"],
            "canonical_argmax_differences_within_1e4": receipt["canonical_argmax_differences_within_1e4"],
        }
    assert set(cells) == {"hk_to_cp_hk", "ukr_to_cp_hk", "hk_to_ukr_rus", "ukr_to_ukr_rus"}
    report = {
        "complete": True,
        "cells": cells,
        "validation": hashes,
        "protocol": {
            "queries_per_cell": 61_440,
            "episodes_per_cell": 512,
            "same_inputs_within_target": True,
            "training": False,
            "encoding": False,
            "sampling": False,
            "gpu": False,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
