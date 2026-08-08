#!/usr/bin/env python3
"""Build the deduplicated fixed-test grid for final-core evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core_plan import ORDERS, SOURCES, CoreModel, build_models


SEEDS = (0, 1, 2)
CHECKPOINT_STEP = 2500
EPISODE_COUNT = 512
PROTOCOL = "fixed_test_512_static_test_on_static_train_v1"


@dataclass(frozen=True)
class PhysicalJob:
    seed: int
    model: CoreModel

    @property
    def key(self) -> tuple[int, str]:
        return self.seed, self.model.model_id


def physical_jobs() -> list[PhysicalJob]:
    jobs = [PhysicalJob(seed, model) for model in build_models() for seed in SEEDS]
    if len(jobs) != 93 or len({job.key for job in jobs}) != 93:
        raise AssertionError("final-core physical plan must contain 93 unique checkpoints")
    return jobs


def checkpoint_path(state_root: Path, job: PhysicalJob, run_stamp: str) -> Path:
    return (
        state_root
        / f"finalcore_{job.model.model_id}_s{job.seed}_{run_stamp}"
        / "checkpoint"
        / f"state_dict_{CHECKPOINT_STEP}.ckpt"
    )


def model_for_ladder(order: str, rung: int) -> CoreModel:
    wanted = frozenset(ORDERS[order][:rung])
    matches = [model for model in build_models() if frozenset(model.sources) == wanted]
    if len(matches) != 1:
        raise AssertionError(f"expected one physical model for ladder:{order}:{rung}")
    return matches[0]


def expected_counts() -> dict[str, int]:
    matrix_checkpoint_seeds = {
        (seed, f"ss_{source}") for seed in SEEDS for source in SOURCES
    }
    ladder_checkpoint_seeds = {
        (seed, model_for_ladder(order, rung).model_id)
        for seed in SEEDS
        for order in ORDERS
        for rung in range(1, 10)
    }
    overlap_checkpoint_seeds = matrix_checkpoint_seeds & ladder_checkpoint_seeds
    union_checkpoint_seeds = matrix_checkpoint_seeds | ladder_checkpoint_seeds
    counts = {
        "matrix_checkpoint_seeds": len(matrix_checkpoint_seeds),
        "ladder_checkpoint_seeds": len(ladder_checkpoint_seeds),
        "overlap_checkpoint_seeds": len(overlap_checkpoint_seeds),
        "union_checkpoint_seeds": len(union_checkpoint_seeds),
        "matrix_cells": len(matrix_checkpoint_seeds) * len(SOURCES),
        "ladder_physical_cells": len(ladder_checkpoint_seeds) * len(SOURCES),
        "overlap_cells": len(overlap_checkpoint_seeds) * len(SOURCES),
        "union_cells": len(union_checkpoint_seeds) * len(SOURCES),
        "ladder_reported_rows": len(SEEDS) * len(ORDERS) * 9 * len(SOURCES),
    }
    expected = {
        "matrix_checkpoint_seeds": 27,
        "ladder_checkpoint_seeds": 75,
        "overlap_checkpoint_seeds": 9,
        "union_checkpoint_seeds": 93,
        "matrix_cells": 243,
        "ladder_physical_cells": 675,
        "overlap_cells": 81,
        "union_cells": 837,
        "ladder_reported_rows": 729,
    }
    if counts != expected:
        raise AssertionError(f"unexpected fixed-test counts: {counts}")
    return counts


def main() -> int:
    expected_counts()
    print("job_index\tseed\tmodel_id\tn_sources\tsources\taliases")
    for index, job in enumerate(physical_jobs()):
        print(
            f"{index}\t{job.seed}\t{job.model.model_id}\t{len(job.model.sources)}\t"
            f"{','.join(job.model.sources)}\t{','.join(job.model.aliases)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
