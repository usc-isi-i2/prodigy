#!/usr/bin/env python3
"""Fail fast when an n_hop=2 saturation config drifts from the protocol."""

from __future__ import annotations

import argparse
from pathlib import Path


HERE = Path(__file__).resolve().parent
COMMON = {
    "task_name": "neighbor_matching",
    "edge_view": "default",
    "feature_subset": "all",
    "original_features": True,
    "layers": "S,U,M",
    "gnn_type": "sage",
    "emb_dim": 256,
    "n_layer": 1,
    "dropout": 0,
    "n_way": 30,
    "n_shots": 3,
    "n_query": 4,
    "batch_size": 1,
    "n_hop": 2,
    "dataset_len_cap": 10_000,
    "val_len_cap": 500,
    "test_len_cap": 500,
    "epochs": 4,
    "eval_step": 100_000,
    "checkpoint_step": 10_000,
    "checkpoint_steps": "0,100,500,1000,2000,10000,40000",
    "workers": 2,
    "seed": 0,
}
ARM_FIELDS = {
    "all8": {
        "dataset": "covid19_twitter",
        "root": "/dataMeR1/phil/data/merged/graphs",
        "graph_filename": "ukr_rus_covid_midterm_all8_retweet_graph.pt",
        "prefix": "sat_h2_all8",
    },
    "ukr": {
        "dataset": "ukr_rus_twitter",
        "root": "/dataMeR1/phil/data/ukr_rus_twitter/graphs",
        "graph_filename": "retweet_graph_parquet.pt",
        "prefix": "sat_h2_ukr",
    },
    "covid": {
        "dataset": "covid19_twitter",
        "root": "/dataMeR1/phil/data/covid19_twitter/graphs",
        "graph_filename": "retweet_graph_parquet.pt",
        "prefix": "sat_h2_covid",
    },
}


def load_simple_yaml(path: Path) -> dict[str, object]:
    payload: dict[str, object] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            raise ValueError(f"{path}:{line_number}: expected key: value")
        key, value = (part.strip() for part in stripped.split(":", 1))
        lowered = value.lower()
        if lowered in {"true", "false"}:
            parsed: object = lowered == "true"
        elif value.startswith('"') and value.endswith('"'):
            parsed = value[1:-1]
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
    actual = sorted(path.name for path in config_dir.glob("train_*.yaml"))
    expected = [f"train_{arm}.yaml" for arm in sorted(ARM_FIELDS)]
    if actual != expected:
        errors.append(f"config set differs: expected={expected}, actual={actual}")

    for arm, arm_fields in ARM_FIELDS.items():
        path = config_dir / f"train_{arm}.yaml"
        if not path.is_file():
            continue
        payload = load_simple_yaml(path)
        for key, expected_value in {**COMMON, **arm_fields}.items():
            if payload.get(key) != expected_value:
                errors.append(
                    f"{path.name}: {key}={payload.get(key)!r}, expected {expected_value!r}"
                )
        if arm == "all8":
            if payload.get("neighbor_sampling_episode_source") != "graph_id":
                errors.append("train_all8.yaml: episodes must be confined by graph_id")
            if payload.get("neighbor_sampling_episode_source_weighting") != "balanced":
                errors.append("train_all8.yaml: source weighting must be balanced")
        else:
            for key in (
                "neighbor_sampling_episode_source",
                "neighbor_sampling_episode_source_weighting",
                "neighbor_sampling_source_subset",
            ):
                if key in payload:
                    errors.append(f"{path.name}: single-source arm must not set {key}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=HERE)
    args = parser.parse_args()
    errors = validate(args.config_dir)
    if errors:
        print("n_hop=2 saturation config validation FAILED:")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("n_hop=2 saturation config validation passed (3 arms, fresh matched-40k).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
