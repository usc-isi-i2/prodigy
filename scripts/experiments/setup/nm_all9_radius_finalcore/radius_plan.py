#!/usr/bin/env python3
"""Immutable registry for the all-nine final-core radius experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
CHECKPOINT_STEPS = (100, 300, 900, 2500)


@dataclass(frozen=True)
class Arm:
    arm_id: str
    config: Path
    radii: tuple[str, ...]


@dataclass(frozen=True)
class Panel:
    panel_id: str
    radii: tuple[str, ...]
    source_confined: bool = False
    primary: bool = True


ARMS = (
    Arm("global", HERE / "global.yaml", ("global",)),
    Arm("radius_mix", HERE / "radius_mix.yaml", ("2", "3", "global")),
    Arm("close_only", HERE / "close_only.yaml", ("2", "3")),
)

PANELS = (
    Panel("radius2", ("2",)),
    Panel("radius3", ("3",)),
    Panel("global", ("global",)),
    Panel("within_source", (), source_confined=True, primary=False),
)


def get_arm(arm_id: str) -> Arm:
    matches = [arm for arm in ARMS if arm.arm_id == arm_id]
    if len(matches) != 1:
        raise ValueError(f"unknown radius arm {arm_id!r}")
    return matches[0]


def get_panel(panel_id: str) -> Panel:
    matches = [panel for panel in PANELS if panel.panel_id == panel_id]
    if len(matches) != 1:
        raise ValueError(f"unknown radius evaluation panel {panel_id!r}")
    return matches[0]


def select_validation_checkpoint(
    rows: list[dict[str, Any]],
    checkpoint_steps: tuple[int, ...] = CHECKPOINT_STEPS,
) -> dict[str, Any]:
    """Select the best step by the macro mean over the three primary panels."""
    primary_ids = {panel.panel_id for panel in PANELS if panel.primary}
    expected = {
        (step, panel_id)
        for step in checkpoint_steps
        for panel_id in primary_ids
    }
    observed = {
        (int(row["checkpoint_step"]), str(row["panel"])) for row in rows
    }
    if len(rows) != len(expected) or observed != expected:
        raise ValueError(
            "validation rows must contain every checkpoint x primary-panel cell exactly once"
        )

    summaries = []
    for step in checkpoint_steps:
        panel_rows = [row for row in rows if int(row["checkpoint_step"]) == step]
        scores = {str(row["panel"]): float(row["score"]) for row in panel_rows}
        summaries.append(
            {
                "checkpoint_step": step,
                "macro_score": sum(scores.values()) / len(primary_ids),
                "panel_scores": scores,
            }
        )
    selected = max(
        summaries,
        key=lambda row: (float(row["macro_score"]), -int(row["checkpoint_step"])),
    )
    return {"selected": selected, "checkpoint_summaries": summaries}


def main() -> int:
    print("arm_id\tconfig\tradii")
    for arm in ARMS:
        print(f"{arm.arm_id}\t{arm.config}\t{','.join(arm.radii)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
