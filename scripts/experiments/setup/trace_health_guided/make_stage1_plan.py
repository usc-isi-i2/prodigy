#!/usr/bin/env python3
"""Generate matched-budget specialist and naive-merge training arms."""

from __future__ import annotations

import argparse
from dataclasses import dataclass


SOURCES = ("ukr_rus", "covid", "midterm", "covid_political")
DEFAULT_SEEDS = (0, 1, 2)


@dataclass(frozen=True)
class Arm:
    model_id: str
    condition: str
    seed: int
    sources: tuple[str, ...]


def parse_values(spec: str) -> tuple[str, ...]:
    return tuple(value.strip() for value in spec.split(",") if value.strip())


def parse_ints(spec: str) -> tuple[int, ...]:
    return tuple(int(value) for value in parse_values(spec))


def build_plan(
    *, sources: tuple[str, ...] = SOURCES,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> list[Arm]:
    if not sources or not set(sources) <= set(SOURCES):
        raise ValueError(f"sources must be a non-empty subset of {SOURCES}")
    if len(set(sources)) != len(sources):
        raise ValueError("sources must be unique")
    if not seeds or any(seed < 0 for seed in seeds):
        raise ValueError("seeds must be nonnegative")
    arms = [
        Arm(
            model_id=f"ss_{source}_s{seed}",
            condition="single",
            seed=seed,
            sources=(source,),
        )
        for seed in seeds
        for source in sources
    ]
    arms.extend(
        Arm(
            model_id=f"merged_s{seed}",
            condition="merged",
            seed=seed,
            sources=sources,
        )
        for seed in seeds
    )
    return arms


def emit(arms: list[Arm]) -> None:
    print("model_id\tcondition\tseed\tsources")
    for arm in arms:
        print(
            f"{arm.model_id}\t{arm.condition}\t{arm.seed}\t"
            f"{','.join(arm.sources)}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", default=",".join(SOURCES))
    parser.add_argument("--seeds", default="0,1,2")
    args = parser.parse_args()
    emit(
        build_plan(
            sources=parse_values(args.sources),
            seeds=parse_ints(args.seeds),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
