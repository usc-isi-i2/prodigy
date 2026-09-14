from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected", type=int, default=56)
    args = parser.parse_args()
    files = sorted(Path(args.raw_root).glob("*.json"))
    if len(files) != args.expected:
        raise RuntimeError(f"expected {args.expected} result files, found {len(files)}")
    rows = [json.loads(path.read_text()) for path in files]
    keys = [
        "target", "sources", "seed", "checkpoint_step", "roc_auc_ovr_macro",
        "f1_macro", "accuracy", "classes", "support_nodes", "test_nodes", "checkpoint",
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            row = dict(row); row["sources"] = ",".join(row["sources"])
            writer.writerow({key: row[key] for key in keys})
    print(f"wrote {output}: {len(rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

