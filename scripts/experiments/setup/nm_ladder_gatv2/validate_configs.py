#!/usr/bin/env python3
"""Fail fast when a GATv2 ladder config drifts from the registered protocol."""

from __future__ import annotations

import argparse
from pathlib import Path


HERE = Path(__file__).resolve().parent
COMMON = {
    "task_name": "neighbor_matching",
    "edge_view": "default",
    "feature_subset": "all",
    "original_features": True,
    "gnn_type": "gat",
    "layers": "S,U,M",
    "emb_dim": 256,
    "n_layer": 1,
    "dropout": 0,
    "n_way": 30,
    "n_shots": 3,
    "n_query": 4,
    "batch_size": 1,
    "n_hop": 1,
    "dataset_len_cap": 10_000,
    "val_len_cap": 500,
    "test_len_cap": 500,
    "epochs": 4,
    "eval_step": 100_000,
    "checkpoint_step": 10_000,
    "workers": 16,
    "seed": 0,
}
INPUTS = {
    1: ("ukr_rus_twitter", "/dataMeR1/phil/data/ukr_rus_twitter/graphs", "retweet_graph_parquet.pt"),
    2: ("covid19_twitter", "/dataMeR1/phil/data/merged/graphs", "ukr_rus_covid_retweet_graph.pt"),
    3: ("covid19_twitter", "/dataMeR1/phil/data/merged/graphs", "ukr_rus_covid_midterm_retweet_graph.pt"),
    4: ("covid19_twitter", "/dataMeR1/phil/data/merged/graphs", "ukr_rus_covid_midterm_4src_retweet_graph.pt"),
    5: ("covid19_twitter", "/dataMeR1/phil/data/merged/graphs", "ukr_rus_covid_midterm_5src_retweet_graph.pt"),
    6: ("covid19_twitter", "/dataMeR1/phil/data/merged/graphs", "ukr_rus_covid_midterm_6src_retweet_graph.pt"),
    7: ("covid19_twitter", "/dataMeR1/phil/data/merged/graphs", "ukr_rus_covid_midterm_7src_retweet_graph.pt"),
    8: ("covid19_twitter", "/dataMeR1/phil/data/merged/graphs", "ukr_rus_covid_midterm_all8_retweet_graph.pt"),
}


def load_simple_yaml(path: Path) -> dict[str, object]:
    """Parse the flat scalar/list subset used by these self-contained configs.

    Keeping this validator stdlib-only lets DRY_RUN work on the laptop. The actual
    trainer still parses the same files with PyYAML in the prodigy environment.
    """
    payload: dict[str, object] = {}
    list_key: str | None = None
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("- "):
            if list_key is None:
                raise ValueError(f"{path}:{line_number}: list item without a key")
            value = stripped[2:].strip()
            assert isinstance(payload[list_key], list)
            payload[list_key].append(value)
            continue
        if ":" not in stripped:
            raise ValueError(f"{path}:{line_number}: expected key: value")
        key, value = (part.strip() for part in stripped.split(":", 1))
        if not value:
            payload[key] = []
            list_key = key
            continue
        list_key = None
        lowered = value.lower()
        if lowered in {"true", "false"}:
            parsed: object = lowered == "true"
        else:
            try:
                parsed = int(value)
            except ValueError:
                try:
                    parsed = float(value)
                except ValueError:
                    parsed = value
        payload[key] = parsed
    return payload


def validate(config_dir: Path = HERE) -> list[str]:
    errors: list[str] = []
    actual = sorted(path.name for path in config_dir.glob("train_*src.yaml"))
    expected = [f"train_{rung}src.yaml" for rung in INPUTS]
    if actual != expected:
        errors.append(f"config set differs: expected={expected}, actual={actual}")

    for rung, (dataset, root, filename) in INPUTS.items():
        path = config_dir / f"train_{rung}src.yaml"
        if not path.is_file():
            continue
        payload = load_simple_yaml(path)
        for key, expected_value in COMMON.items():
            if payload.get(key) != expected_value:
                errors.append(
                    f"{path.name}: {key}={payload.get(key)!r}, expected {expected_value!r}"
                )
        for key, expected_value in {
            "dataset": dataset,
            "root": root,
            "graph_filename": filename,
            "prefix": f"nm_ladder_gatv2_r{rung}_{rung}src",
        }.items():
            if payload.get(key) != expected_value:
                errors.append(
                    f"{path.name}: {key}={payload.get(key)!r}, expected {expected_value!r}"
                )
        tags = payload.get("tags") or []
        for tag in ("nm_ladder_gatv2", f"rung_{rung}"):
            if tag not in tags:
                errors.append(f"{path.name}: missing tag {tag!r}")
        if rung == 1:
            for key in (
                "neighbor_sampling_episode_source",
                "neighbor_sampling_episode_source_weighting",
            ):
                if key in payload:
                    errors.append(f"{path.name}: single-source rung must not set {key}")
        else:
            if payload.get("neighbor_sampling_episode_source") != "graph_id":
                errors.append(f"{path.name}: merged rung is not confined by graph_id")
            if payload.get("neighbor_sampling_episode_source_weighting") != "balanced":
                errors.append(f"{path.name}: merged rung is not source-balanced")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=HERE)
    args = parser.parse_args()
    errors = validate(args.config_dir)
    if errors:
        print("GATv2 ladder config validation FAILED:")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("GATv2 ladder config validation passed (8 rungs, matched-40k).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
