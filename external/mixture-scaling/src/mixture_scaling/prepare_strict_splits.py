from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .strict_data import load_raw, save_split, split_summary, stratified_node_split


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    output_root = Path(args.output_root)
    summaries = []
    for name, graph_config in config["graphs"].items():
        raw = load_raw(graph_config["path"])
        split = stratified_node_split(raw.get("y"), int(raw["x"].shape[0]), int(config["protocol"]["split_seed"]))
        path = output_root / f"{name}.pt"
        if path.exists():
            raise FileExistsError(f"split already exists: {path}")
        save_split(path, split, name, int(raw["x"].shape[0]))
        summaries.append(split_summary(name, split, raw.get("y")))
    (output_root / "manifest.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
