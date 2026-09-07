"""Verify exact input pairing between two completed, trusted local eval exports.

Does not compare model states or predictions, which should differ across seeds.
No artifacts are written; tensor files must come from our trusted experiment runs.
"""
import argparse
import json
from pathlib import Path

import torch
from torch_geometric.data import Data

from .run_native import file_sha256


def inventory(root, episodes):
    root = Path(root).resolve()
    if json.loads((root / "execution_status.json").read_text())["status"] != "complete":
        raise ValueError("Evaluation must be complete")
    records = json.loads((root / "paired_episodes/index.json").read_text())["records"]
    if len(records) != episodes or [r["ordinal"] for r in records] != list(range(episodes)):
        raise ValueError("Unexpected episode inventory")
    paths = []
    for ordinal, record in enumerate(records):
        if record["file"] != f"episode_{ordinal:05d}.pt":
            raise ValueError("Unexpected episode filename")
        path = (root / "paired_episodes" / record["file"]).resolve()
        if not path.is_relative_to(root) or file_sha256(path) != record["sha256"]:
            raise ValueError("Episode receipt mismatch")
        paths.append(path)
    return paths


def equal_value(x, y):
    if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
        return x.dtype == y.dtype and x.shape == y.shape and torch.equal(x, y)
    if isinstance(x, Data) and isinstance(y, Data):
        return equal_value(x.to_dict(), y.to_dict())
    if type(x) is not type(y):
        return False
    if isinstance(x, dict):
        return x.keys() == y.keys() and all(equal_value(x[k], y[k]) for k in x)
    if isinstance(x, (tuple, list)):
        return len(x) == len(y) and all(equal_value(a, b) for a, b in zip(x, y))
    if x is None or isinstance(x, (bool, int, float, str)):
        return x == y
    raise TypeError(f"Unsupported input value: {type(x).__name__}")


def compare_captures(left, right):
    a, b = left["input"], right["input"]
    if len(a) != len(b):
        return ["input_count"]
    differences = []
    for i, (x, y) in enumerate(zip(a, b)):
        if not equal_value(x, y):
            differences.append(f"input_{i}")
    x, y = left["output"]["y_true"], right["output"]["y_true"]
    if x.dtype != y.dtype or x.shape != y.shape or not torch.equal(x, y):
        differences.append("query_truth")
    return differences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=500)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    torch.set_num_threads(1)
    left, right = inventory(args.left, args.episodes), inventory(args.right, args.episodes)
    mismatches = []
    for ordinal, (a, b) in enumerate(zip(left, right)):
        captures = [torch.load(p, map_location="cpu", weights_only=False)["native_capture"]
                    for p in (a, b)]
        differences = compare_captures(*captures)
        if differences:
            mismatches.append(dict(ordinal=ordinal, fields=differences))
    print(json.dumps(dict(episodes=args.episodes, exact_inputs_and_truth=not mismatches,
                         mismatches=mismatches), indent=2))
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
