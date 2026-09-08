#!/usr/bin/env python3
"""Generate the matched cross-graph-ratio, data-scale, and capacity configs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[4]
BASE_CONFIG = (
    ROOT
    / "scripts/experiments/setup/nm_interventions_overnight/configs/baseline_r8_s0.yaml"
)
SEEDS = (0, 1, 2)
CHECKPOINT_STEPS = (2_000, 4_000, 6_000, 8_000, 10_000)


@dataclass(frozen=True)
class Arm:
    name: str
    cross_graph_prob: float
    emb_dim: int = 256


ARMS = (
    Arm("mix_p000", 0.00),
    Arm("mix_p010", 0.10),
    Arm("mix_p025", 0.25),
    Arm("mix_p050", 0.50),
    Arm("mix_p100", 1.00),
    Arm("wide_p000", 0.00, 512),
)


def write_configs(output: Path, base_config: Path = BASE_CONFIG) -> list[Path]:
    base = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    expected_sources = (
        "ukr_rus,covid,midterm,covid_political,election2020,"
        "ukr_rus_suspended,cp_hk,facebook_page_reference"
    )
    if base.get("neighbor_sampling_source_subset") != expected_sources:
        raise ValueError("flagship rung-eight source set drifted")
    if base.get("campaign_holdout") != "twibot20":
        raise ValueError("flagship permanent holdout drifted")
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for arm in ARMS:
        config = dict(base)
        config.update(
            prefix=f"paper_mech_{arm.name}",
            campaign_flags="",
            neighbor_sampling_cross_source_prob=arm.cross_graph_prob,
            emb_dim=arm.emb_dim,
            dataset_len_cap=10_000,
            epochs=1,
            campaign_eval_interval=2_000,
            early_stopping_patience=99,
            checkpoint_step=2_000,
        )
        path = output / f"train_{arm.name}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        paths.append(path)
    (output / "arms.tsv").write_text(
        "arm\tcross_graph_prob\temb_dim\tconfig\n"
        + "".join(
            f"{arm.name}\t{arm.cross_graph_prob:g}\t{arm.emb_dim}\ttrain_{arm.name}.yaml\n"
            for arm in ARMS
        ),
        encoding="utf-8",
    )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, default=BASE_CONFIG)
    args = parser.parse_args()
    paths = write_configs(args.output.resolve(), args.base_config.resolve())
    if len(paths) != len(ARMS):
        raise ValueError("mechanism config coverage mismatch")
    print(f"wrote {len(paths)} configs to {args.output.resolve()}")


if __name__ == "__main__":
    main()
