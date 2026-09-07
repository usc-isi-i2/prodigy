#!/usr/bin/env python3
"""Generate order-only schedule interventions for the TRACE scaling study."""

from __future__ import annotations

import argparse
from dataclasses import dataclass


SOURCES = ("ukr_rus", "covid", "midterm", "covid_political")
SCHEDULES = ("blocked", "replay100", "interleaved")
DEFAULT_RUNGS = (2, 3, 4)
DEFAULT_SEEDS = (0, 1, 2)


@dataclass(frozen=True)
class Arm:
    model_id: str
    rung: int
    seed: int
    schedule: str
    sources: tuple[str, ...]
    order: tuple[str, ...]
    segment_sources: tuple[str, ...]
    segment_steps: tuple[int, ...]
    source_counts: tuple[int, ...]


def parse_ints(spec: str) -> tuple[int, ...]:
    return tuple(int(value) for value in spec.split(",") if value.strip())


def allocate_counts(total_steps: int, source_count: int) -> tuple[int, ...]:
    base, remainder = divmod(total_steps, source_count)
    return tuple(base + int(index < remainder) for index in range(source_count))


def rotate(values: tuple[str, ...], amount: int) -> tuple[str, ...]:
    amount %= len(values)
    return values[amount:] + values[:amount]


def schedule_segments(
    order: tuple[str, ...], counts: dict[str, int], block_size: int
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    remaining = dict(counts)
    segment_sources: list[str] = []
    segment_steps: list[int] = []
    while any(remaining.values()):
        for source in order:
            take = min(block_size, remaining[source])
            if take <= 0:
                continue
            segment_sources.append(source)
            segment_steps.append(take)
            remaining[source] -= take
    return tuple(segment_sources), tuple(segment_steps)


def build_plan(
    *, total_steps: int = 2500, replay_block: int = 100,
    rungs: tuple[int, ...] = DEFAULT_RUNGS,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    schedules: tuple[str, ...] = SCHEDULES,
) -> list[Arm]:
    if total_steps <= 0 or replay_block <= 0:
        raise ValueError("step counts must be positive")
    if not rungs or any(rung not in DEFAULT_RUNGS for rung in rungs):
        raise ValueError(f"rungs must be drawn from {DEFAULT_RUNGS}")
    if not seeds or any(seed < 0 for seed in seeds):
        raise ValueError("seeds must be nonnegative")
    if not schedules or not set(schedules) <= set(SCHEDULES):
        raise ValueError(f"schedules must be drawn from {SCHEDULES}")

    arms = []
    for rung in rungs:
        sources = SOURCES[:rung]
        canonical_counts = allocate_counts(total_steps, rung)
        counts = dict(zip(sources, canonical_counts))
        for seed in seeds:
            order = rotate(sources, seed)
            for schedule in schedules:
                block_size = {
                    "blocked": total_steps,
                    "replay100": replay_block,
                    "interleaved": 1,
                }[schedule]
                segment_sources, segment_steps = schedule_segments(
                    order, counts, block_size
                )
                if sum(segment_steps) != total_steps:
                    raise AssertionError("schedule does not fill the training budget")
                realized = tuple(
                    sum(
                        step for source, step in zip(segment_sources, segment_steps)
                        if source == expected
                    )
                    for expected in sources
                )
                if realized != canonical_counts:
                    raise AssertionError("schedule changed per-source exposure")
                arms.append(
                    Arm(
                        model_id=f"r{rung}_{schedule}_s{seed}",
                        rung=rung,
                        seed=seed,
                        schedule=schedule,
                        sources=sources,
                        order=order,
                        segment_sources=segment_sources,
                        segment_steps=segment_steps,
                        source_counts=canonical_counts,
                    )
                )
    return arms


def emit(arms: list[Arm]) -> None:
    print(
        "model_id\trung\tseed\tschedule\tsources\torder\t"
        "segment_sources\tsegment_steps\tsource_counts"
    )
    for arm in arms:
        print(
            "\t".join(
                (
                    arm.model_id,
                    str(arm.rung),
                    str(arm.seed),
                    arm.schedule,
                    ",".join(arm.sources),
                    ",".join(arm.order),
                    ",".join(arm.segment_sources),
                    ",".join(str(value) for value in arm.segment_steps),
                    ",".join(str(value) for value in arm.source_counts),
                )
            )
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-steps", type=int, default=2500)
    parser.add_argument("--replay-block", type=int, default=100)
    parser.add_argument("--rungs", default="2,3,4")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--schedules", default=",".join(SCHEDULES))
    args = parser.parse_args()
    emit(
        build_plan(
            total_steps=args.total_steps,
            replay_block=args.replay_block,
            rungs=parse_ints(args.rungs),
            seeds=parse_ints(args.seeds),
            schedules=tuple(
                value.strip() for value in args.schedules.split(",") if value.strip()
            ),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
