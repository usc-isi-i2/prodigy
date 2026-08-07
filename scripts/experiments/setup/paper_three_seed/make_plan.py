#!/usr/bin/env python3
"""Print the registered 57-condition PRODIGY paper replication plan as TSV."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


HERE = Path(__file__).resolve().parent
SETUP = HERE.parent


@dataclass(frozen=True)
class Job:
    family: str
    arm: str
    config: Path
    eval_group: str
    target_step: int


SPECIALISTS = (
    "ukr_rus_twitter",
    "covid19_twitter",
    "midterm",
    "covid_political",
    "election2020",
    "ukr_rus_suspended",
    "twibot20",
    "cp_hk_twitter",
)


def job(family: str, arm: str, config: Path, eval_group: str, target_step: int = 40_000) -> Job:
    if not config.is_file():
        raise FileNotFoundError(config)
    return Job(family, arm, config, eval_group, target_step)


def build_plan() -> list[Job]:
    jobs: list[Job] = []
    single_root = SETUP / "nm_single_source_matrix"
    for source in SPECIALISTS:
        jobs.append(job("specialist", source, single_root / f"{source}.yaml", "sage_1hop"))

    ladder_root = SETUP / "nm_ladder_nhop2" / "configs"
    ladder_names = [
        *(f"train_ordA_r{r}.yaml" for r in range(2, 9)),
        *(f"train_ordB_r{r}.yaml" for r in range(3, 8)),
        *(f"train_ordC_r{r}.yaml" for r in range(2, 8)),
    ]
    for name in ladder_names:
        jobs.append(job("ladder_1hop", Path(name).stem.removeprefix("train_"), ladder_root / name, "sage_1hop"))

    for rung in range(1, 9):
        name = f"train_ordA_r{rung}.yaml"
        jobs.append(job("ladder_2hop", f"ordA_r{rung}", ladder_root / name, "sage_2hop"))

    gat_root = SETUP / "nm_ladder_gatv2"
    for rung in range(1, 9):
        jobs.append(job("ladder_gatv2", f"ordA_r{rung}", gat_root / f"train_{rung}src.yaml", "gat_1hop"))

    fixed_root = SETUP / "nm_ladder_fixed_exposure_nhop2" / "configs"
    fixed_names = [
        *(f"train_ordA_r{r}.yaml" for r in range(1, 9)),
        *(f"train_ordC_r{r}.yaml" for r in range(1, 8)),
    ]
    for name in fixed_names:
        rung = int(Path(name).stem.rsplit("r", 1)[1])
        jobs.append(job(
            "fixed_exposure_2hop",
            Path(name).stem.removeprefix("train_"),
            fixed_root / name,
            "sage_2hop",
            10_000 * rung,
        ))

    keys = [(item.family, item.arm) for item in jobs]
    if len(jobs) != 57 or len(set(keys)) != len(keys):
        raise AssertionError(f"Expected 57 unique jobs, got {len(jobs)} ({len(set(keys))} unique).")
    counts = {group: sum(item.eval_group == group for item in jobs) for group in {item.eval_group for item in jobs}}
    if counts != {"sage_1hop": 26, "gat_1hop": 8, "sage_2hop": 23}:
        raise AssertionError(f"Unexpected evaluation groups: {counts}")
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--absolute", action="store_true", help="Print absolute config paths.")
    args = parser.parse_args()
    repo_root = HERE.parents[3]
    print("family\tarm\tconfig\teval_group\ttarget_step")
    for item in build_plan():
        config = item.config if args.absolute else item.config.relative_to(repo_root)
        print(f"{item.family}\t{item.arm}\t{config}\t{item.eval_group}\t{item.target_step}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
