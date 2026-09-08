#!/usr/bin/env python3
"""Generate the predeclared 2x3x3 signal-first LOO schedule experiment."""
from __future__ import annotations

import json
import random
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "nm_leave_one_out_finalcore" / "training.yaml"
OUT = HERE / "configs"
SOURCES = {
    "covid": 23_012_850,
    "midterm": 341_908,
    "covid_political": 78_672,
    "election2020": 78_932,
    "ukr_rus_suspended": 72_295,
    "twibot20": 162_990,
    "cp_hk": 333_800,
    "facebook_page_reference": 150_000,
}
TOTAL_STEPS = 2500


def apportion(weights: dict[str, int]) -> dict[str, int]:
    total = sum(weights.values())
    raw = {name: TOTAL_STEPS * value / total for name, value in weights.items()}
    counts = {name: int(value) for name, value in raw.items()}
    for name in sorted(raw, key=lambda x: raw[x] - counts[x], reverse=True)[: TOTAL_STEPS - sum(counts.values())]:
        counts[name] += 1
    return counts


def segments(exposure: str, block: str, seed: int) -> tuple[list[str], list[int]]:
    weights = {name: 1 for name in SOURCES} if exposure == "uniform" else SOURCES
    counts = apportion(weights)
    rng = random.Random(70_000 + seed)
    if block == "blocked":
        order = list(SOURCES)
        rng.shuffle(order)
        return order, [counts[name] for name in order]
    width = int(block)
    names: list[str] = []
    steps: list[int] = []
    remaining = counts.copy()
    while sum(remaining.values()):
        choices = [name for name, count in remaining.items() if count]
        name = rng.choices(choices, weights=[remaining[x] for x in choices], k=1)[0]
        take = min(width, remaining[name])
        if names and names[-1] == name:
            steps[-1] += take
        else:
            names.append(name); steps.append(take)
        remaining[name] -= take
    return names, steps


def main() -> None:
    base = yaml.safe_load(BASE.read_text())
    OUT.mkdir(exist_ok=True)
    manifest = []
    for seed in range(3):
        for exposure in ("uniform", "proportional"):
            for block in ("1", "16", "blocked"):
                names, steps = segments(exposure, block, seed)
                model_id = f"loo_ukr_{exposure}_k{block}_s{seed}"
                cfg = dict(base)
                cfg.update({
                    "batch_size": 1,
                    "seed": seed,
                    "prefix": model_id,
                    "neighbor_sampling_source_subset": ",".join(SOURCES),
                    "neighbor_sampling_source_schedule": ",".join(names),
                    "neighbor_sampling_source_schedule_steps": ",".join(map(str, steps)),
                    "neighbor_sampling_source_schedule_seed": seed,
                    "neighbor_matching_member_seed": seed,
                    "neighbor_sampling_episode_source_weighting": "balanced" if exposure == "uniform" else "proportional",
                    "tags": ["nm_loo_schedule_signal", "holdout_ukr_rus", exposure, f"k_{block}", f"seed_{seed}"],
                })
                path = OUT / f"{model_id}.yaml"
                path.write_text(yaml.safe_dump(cfg, sort_keys=False))
                manifest.append({"model_id": model_id, "seed": seed, "exposure": exposure,
                                 "block_size": block, "counts": apportion({n: 1 for n in SOURCES} if exposure == "uniform" else SOURCES),
                                 "config": str(path.relative_to(HERE))})
    (HERE / "plan.json").write_text(json.dumps(manifest, indent=2) + "\n")
    lines = ["model_id\tseed\texposure\tblock_size\tconfig"]
    lines.extend(
        f"{row['model_id']}\t{row['seed']}\t{row['exposure']}\t{row['block_size']}\t{row['config']}"
        for row in manifest
    )
    (HERE / "plan.tsv").write_text("\n".join(lines) + "\n")
    assert len(manifest) == 18


if __name__ == "__main__":
    main()
