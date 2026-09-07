#!/usr/bin/env python3
"""Registered targets and leave-one-family-out pretraining source sets for RQ1."""

from __future__ import annotations

from dataclasses import dataclass


ALL_SOURCES = (
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


@dataclass(frozen=True)
class TargetPlan:
    target: str
    excluded_family: tuple[str, ...]

    @property
    def sources(self) -> tuple[str, ...]:
        return tuple(source for source in ALL_SOURCES if source not in self.excluded_family)


TARGETS = (
    TargetPlan("covid_political", ("covid", "covid_political")),
    TargetPlan("election2020", ("election2020",)),
    TargetPlan("ukr_rus_suspended", ("ukr_rus", "ukr_rus_suspended")),
    TargetPlan("twibot20", ("twibot20",)),
)


def main() -> int:
    print("target\texcluded_family\tsources")
    for row in TARGETS:
        print(f"{row.target}\t{','.join(row.excluded_family)}\t{','.join(row.sources)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
