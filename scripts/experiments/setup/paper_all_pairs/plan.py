#!/usr/bin/env python3
"""Build every two-source subset under the final-core training protocol."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import itertools
from pathlib import Path
import re

import yaml


ROOT = Path(__file__).resolve().parents[4]
BASE_CONFIG = ROOT / "scripts/experiments/setup/final_core/training.yaml"
SOURCES = (
    "ukr_rus",
    "covid",
    "midterm",
    "covid_political",
    "election2020",
    "ukr_rus_suspended",
    "twibot20",
    "cp_hk",
    "facebook_page_reference",
)
SOURCE_TAGS = {
    "ukr_rus": "ukr",
    "covid": "covid",
    "midterm": "midterm",
    "covid_political": "covpol",
    "election2020": "election",
    "ukr_rus_suspended": "ukrsusp",
    "twibot20": "twibot",
    "cp_hk": "hongkong",
    "facebook_page_reference": "facebook",
}
STEPS = 2500
SEEDS = (0, 1, 2)


@dataclass(frozen=True)
class Pair:
    left: str
    right: str

    @property
    def sources(self) -> tuple[str, str]:
        return self.left, self.right

    @property
    def arm(self) -> str:
        return f"{SOURCE_TAGS[self.left]}__{SOURCE_TAGS[self.right]}"

    @property
    def prefix(self) -> str:
        return f"paper_pair_{self.arm}"


def build_pairs() -> list[Pair]:
    return [Pair(left, right) for left, right in itertools.combinations(SOURCES, 2)]


def write_configs(output: Path, base_config: Path = BASE_CONFIG) -> list[Path]:
    base = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    if int(base["dataset_len_cap"]) * int(base["epochs"]) != STEPS:
        raise ValueError("final-core base no longer has the registered 2,500-update budget")
    if base.get("graph_filename") != "ukr_rus_covid_midterm_all9_facebook_final_core_split_seed0.pt":
        raise ValueError("final-core base no longer points to the registered all-nine graph")
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for pair in build_pairs():
        config = dict(base)
        config["neighbor_sampling_source_subset"] = ",".join(pair.sources)
        config["prefix"] = pair.prefix
        path = output / f"train_{pair.arm}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        paths.append(path)
    manifest = output / "pairs.tsv"
    manifest.write_text(
        "arm\tleft\tright\tprefix\tconfig\n"
        + "".join(
            f"{pair.arm}\t{pair.left}\t{pair.right}\t{pair.prefix}\ttrain_{pair.arm}.yaml\n"
            for pair in build_pairs()
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
    if len(paths) != 36 or any(not re.fullmatch(r"train_[a-z0-9_]+\.yaml", path.name) for path in paths):
        raise ValueError("pair config generation failed its exact coverage gate")
    print(f"wrote {len(paths)} pair configs to {args.output.resolve()}")


if __name__ == "__main__":
    main()
