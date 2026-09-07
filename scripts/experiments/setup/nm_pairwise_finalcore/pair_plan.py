#!/usr/bin/env python3
"""Canonical one-seed plan for every unordered pair of final-core sources."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
import json
from pathlib import Path


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
SEEDS = (0,)
CHECKPOINT_STEP = 2500
EPISODE_COUNT = 512
PROTOCOL = "fixed_test_512_static_test_on_static_train_v1"


@dataclass(frozen=True)
class PairModel:
    model_id: str
    sources: tuple[str, str]
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class PairJob:
    seed: int
    model: PairModel

    @property
    def key(self) -> tuple[int, str]:
        return self.seed, self.model.model_id


def model_id(left: str, right: str) -> str:
    return f"nmpair_{left}__{right}"


def build_models() -> list[PairModel]:
    models = [
        PairModel(
            model_id(left, right),
            (left, right),
            (f"pair:{left}+{right}",),
        )
        for left, right in combinations(SOURCES, 2)
    ]
    if len(models) != 36 or len({frozenset(model.sources) for model in models}) != 36:
        raise AssertionError("pair plan must contain all 36 unordered source pairs")
    return models


def physical_jobs() -> list[PairJob]:
    jobs = [PairJob(seed, model) for seed in SEEDS for model in build_models()]
    if len(jobs) != 36 or len({job.key for job in jobs}) != 36:
        raise AssertionError("one-seed pair plan must contain 36 checkpoints")
    return jobs


@lru_cache(maxsize=None)
def _checkpoint_map(run_dir_text: str) -> dict[str, Path]:
    run_dir = Path(run_dir_text)
    payload = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    mapping: dict[str, Path] = {}
    for row in payload.get("jobs", []):
        prefix = str(row.get("prefix", ""))
        exp_name = str(row.get("exp_name", ""))
        if prefix.startswith("nmpair_") and exp_name:
            mapping[prefix] = (
                run_dir / "state" / exp_name / "checkpoint"
                / f"state_dict_{CHECKPOINT_STEP}.ckpt"
            )
    expected = {model.model_id for model in build_models()}
    if set(mapping) != expected:
        missing = sorted(expected - set(mapping))
        extra = sorted(set(mapping) - expected)
        raise ValueError(
            f"shared-run manifest pair mismatch: missing={missing[:3]} extra={extra[:3]}"
        )
    return mapping


def checkpoint_path(run_dir: Path, job: PairJob, _run_stamp: str) -> Path:
    """Resolve a pair checkpoint from the shared trainer's immutable manifest."""
    return _checkpoint_map(str(run_dir.resolve()))[job.model.model_id]


def main() -> int:
    print("model_id\tseed\tsource_left\tsource_right\tsources")
    for job in physical_jobs():
        left, right = job.model.sources
        print(f"{job.model.model_id}\t{job.seed}\t{left}\t{right}\t{left},{right}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
