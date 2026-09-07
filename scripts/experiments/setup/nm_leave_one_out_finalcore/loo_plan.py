#!/usr/bin/env python3
"""Canonical seed-zero leave-one-source-out plan for final-core NM."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
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
class LooModel:
    model_id: str
    sources: tuple[str, ...]
    heldout: str
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class LooJob:
    seed: int
    model: LooModel

    @property
    def key(self) -> tuple[int, str]:
        return self.seed, self.model.model_id


def model_id(heldout: str) -> str:
    return f"nmloo_without_{heldout}"


def build_models() -> list[LooModel]:
    models = [
        LooModel(
            model_id(heldout),
            tuple(source for source in SOURCES if source != heldout),
            heldout,
            (f"loo:without:{heldout}",),
        )
        for heldout in SOURCES
    ]
    if len(models) != 9:
        raise AssertionError("leave-one-out plan must contain nine models")
    for model in models:
        if len(model.sources) != 8 or model.heldout in model.sources:
            raise AssertionError(f"invalid leave-one-out model {model.model_id}")
    return models


def physical_jobs() -> list[LooJob]:
    jobs = [LooJob(seed, model) for seed in SEEDS for model in build_models()]
    if len(jobs) != 9 or len({job.key for job in jobs}) != 9:
        raise AssertionError("seed-zero leave-one-out plan must contain nine checkpoints")
    return jobs


@lru_cache(maxsize=None)
def _checkpoint_map(run_dir_text: str) -> dict[str, Path]:
    run_dir = Path(run_dir_text)
    payload = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    mapping: dict[str, Path] = {}
    for row in payload.get("jobs", []):
        prefix = str(row.get("prefix", ""))
        exp_name = str(row.get("exp_name", ""))
        if prefix.startswith("nmloo_without_") and exp_name:
            mapping[prefix] = (
                run_dir / "state" / exp_name / "checkpoint"
                / f"state_dict_{CHECKPOINT_STEP}.ckpt"
            )
    expected = {model.model_id for model in build_models()}
    if set(mapping) != expected:
        missing = sorted(expected - set(mapping))
        extra = sorted(set(mapping) - expected)
        raise ValueError(
            f"shared-run manifest LOO mismatch: missing={missing[:3]} extra={extra[:3]}"
        )
    return mapping


def checkpoint_path(run_dir: Path, job: LooJob, _run_stamp: str) -> Path:
    """Resolve a LOO checkpoint from the shared trainer's immutable manifest."""
    return _checkpoint_map(str(run_dir.resolve()))[job.model.model_id]


def main() -> int:
    print("model_id\tseed\theldout\tsources")
    for job in physical_jobs():
        print(
            f"{job.model.model_id}\t{job.seed}\t{job.model.heldout}\t"
            f"{','.join(job.model.sources)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
