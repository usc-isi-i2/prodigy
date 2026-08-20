#!/usr/bin/env python3
"""Evaluate the seed-0 step-100 ladder on the final-core fixed NM streams.

This is a thin registered-plan adapter around final_core.evaluate_fixed_grid.  It
changes only the checkpoint inventory, path convention, step label, and protocol
name.  Episode construction, fingerprints, metrics, and replay behavior remain the
final-core implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from scripts.experiments.setup.final_core.core_plan import ORDERS, CoreModel, build_models


CHECKPOINT_STEP = 100
PROTOCOL = "fixed_nm_512_static_test_on_static_train_step100_v1"
RUN_STAMP = "20260810"


@dataclass(frozen=True)
class Job:
    seed: int
    model: CoreModel

    @property
    def key(self) -> tuple[int, str]:
        return self.seed, self.model.model_id


def model_for_ladder(order: str, rung: int) -> CoreModel:
    wanted = frozenset(ORDERS[order][:rung])
    matches = [model for model in build_models() if frozenset(model.sources) == wanted]
    if len(matches) != 1:
        raise AssertionError(f"expected one physical model for {order}/rung {rung}")
    return matches[0]


def ladder_jobs() -> list[Job]:
    models = {
        model_for_ladder(order, rung).model_id: model_for_ladder(order, rung)
        for order in ORDERS
        for rung in range(1, 10)
    }
    jobs = [Job(seed=0, model=models[model_id]) for model_id in sorted(models)]
    if len(jobs) != 25 or len({job.key for job in jobs}) != 25:
        raise AssertionError(f"expected 25 physical ladder checkpoints, got {len(jobs)}")
    return jobs


def checkpoint_path(state_root: Path, job: Job, run_stamp: str) -> Path:
    if job.seed != 0:
        raise ValueError("the architecture matrix registered only training seed 0")
    return (
        state_root
        / "prodigy"
        / f"archmatrix_prodigy_{job.model.model_id}_s0_{run_stamp}"
        / "checkpoint"
        / f"state_dict_{CHECKPOINT_STEP}.ckpt"
    )


def main() -> int:
    from scripts.experiments.setup.final_core import evaluate_fixed_grid as fixed

    # All uses are runtime global lookups inside final_core.evaluate_fixed_grid.
    fixed.CHECKPOINT_STEP = CHECKPOINT_STEP
    fixed.PROTOCOL = PROTOCOL
    fixed.physical_jobs = ladder_jobs
    fixed.checkpoint_path = checkpoint_path
    return fixed.main()


if __name__ == "__main__":
    raise SystemExit(main())
