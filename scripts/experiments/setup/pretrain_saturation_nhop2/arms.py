"""Registered arms and checkpoint keys for the fresh n_hop=2 saturation rerun."""

from __future__ import annotations

from dataclasses import dataclass


STEPS = (0, 100, 500, 1_000, 2_000, 10_000, 40_000)
TRAINED_STEPS = tuple(step for step in STEPS if step > 0)


@dataclass(frozen=True)
class Arm:
    name: str
    config: str
    prefix: str
    corpus: str


ARMS = (
    Arm("all8", "train_all8.yaml", "sat_h2_all8", "8-source merge, within-balanced"),
    Arm("ukr", "train_ukr.yaml", "sat_h2_ukr", "ukr_rus_twitter"),
    Arm("covid", "train_covid.yaml", "sat_h2_covid", "covid19_twitter"),
)
ARMS_BY_NAME = {arm.name: arm for arm in ARMS}


def model_key(arm: str, step: int) -> str:
    if arm not in ARMS_BY_NAME:
        raise ValueError(f"unknown arm: {arm!r}")
    if step not in STEPS:
        raise ValueError(f"unregistered step: {step}")
    return f"sat_h2_{arm}_s{step:06d}"


SHARED_STEP0_KEY = "sat_h2_shared_s000000"
