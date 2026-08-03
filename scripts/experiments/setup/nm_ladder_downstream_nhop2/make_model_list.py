#!/usr/bin/env python3
"""Resolve the completed fair-two-hop ladders into one downstream model list.

The four experiments live in separate Tucker worktrees, so checkpoint discovery must
use four explicit state roots. The emitted row map has 40 logical trajectory rows backed
by 39 physical encoders because fixed-exposure A8 and C8 share one all-eight checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]

DEFAULT_STATE_ROOTS = {
    "matched40k": Path("/dataMeR1/phil/gfm/prodigy-nmlh2/state"),
    "sequential": Path("/dataMeR1/phil/gfm/prodigy-nmlh2seq/state"),
    "split": Path("/dataMeR1/phil/gfm/prodigy-nmlsplit-h2/state"),
    "fixed10k": Path("/dataMeR1/phil/gfm/prodigy-nmlfxh2/state"),
}


@dataclass(frozen=True)
class Variant:
    name: str
    manifest: Path
    orders: tuple[str, ...]
    schedule: str
    exposure: str
    train_edges: str
    default_step: int


VARIANTS = (
    Variant(
        "matched40k",
        REPO_ROOT / "scripts/experiments/setup/nm_ladder_nhop2/manifest.tsv",
        ("A",),
        "interleaved",
        "matched_total_40k",
        "full_adjacency",
        40_000,
    ),
    Variant(
        "sequential",
        REPO_ROOT / "scripts/experiments/setup/nm_ladder_sequential_nhop2/manifest.tsv",
        ("A",),
        "sequential_blocks",
        "matched_total_40k",
        "full_adjacency",
        40_000,
    ),
    Variant(
        "split",
        REPO_ROOT / "scripts/experiments/setup/nm_ladder_train_test_nhop2/manifest.tsv",
        ("A",),
        "interleaved",
        "matched_total_40k",
        "background_train_holdout_nm",
        40_000,
    ),
    Variant(
        "fixed10k",
        REPO_ROOT / "scripts/experiments/setup/nm_ladder_fixed_exposure_nhop2/manifest.tsv",
        ("A", "C"),
        "interleaved",
        "10k_per_active_source",
        "full_adjacency",
        0,
    ),
)


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def logical_rows() -> list[dict[str, object]]:
    """Return the registered 40 logical rows without touching checkpoint state."""
    rows: list[dict[str, object]] = []
    for spec in VARIANTS:
        for raw in _read_manifest(spec.manifest):
            order = raw.get("order") or "A"
            if order not in spec.orders:
                continue
            prefix = raw["model_prefix"]
            step = int(raw.get("target_step") or spec.default_step)
            rung = int(raw["rung"])
            rows.append(
                {
                    "variant": spec.name,
                    "order": order,
                    "rung": rung,
                    "added": raw["added"],
                    "n_sources": int(raw["n_sources"]),
                    "sources": raw["sources"],
                    "schedule": spec.schedule,
                    "exposure": spec.exposure,
                    "train_edges": spec.train_edges,
                    "checkpoint_step": step,
                    "model_prefix": prefix,
                    "model_key": f"{spec.name}__{prefix}",
                    "logical_id": f"{spec.name}_{order}_r{rung}",
                }
            )
    rows.sort(key=lambda row: (
        [spec.name for spec in VARIANTS].index(str(row["variant"])),
        str(row["order"]),
        int(row["rung"]),
    ))
    return rows


def physical_models(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Deduplicate logical rows while rejecting inconsistent reuse metadata."""
    by_key: dict[str, dict[str, object]] = {}
    ordered: list[dict[str, object]] = []
    for row in rows:
        key = str(row["model_key"])
        if key in by_key:
            prior = by_key[key]
            assert prior["checkpoint_step"] == row["checkpoint_step"]
            assert prior["model_prefix"] == row["model_prefix"]
            continue
        by_key[key] = row
        ordered.append(row)
    return ordered


def resolve_checkpoint(state_root: Path, prefix: str, step: int) -> Path | None:
    candidates = sorted(
        (path for path in state_root.glob(f"{prefix}_*") if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for run_dir in candidates:
        checkpoint = run_dir / "checkpoint" / f"state_dict_{step}.ckpt"
        if checkpoint.is_file():
            return checkpoint
    return None


def _state_default(name: str) -> Path:
    env = os.environ.get(f"{name.upper()}_STATE_ROOT")
    return Path(env) if env else DEFAULT_STATE_ROOTS[name]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in DEFAULT_STATE_ROOTS:
        parser.add_argument(
            f"--{name}-state-root",
            type=Path,
            default=_state_default(name),
        )
    parser.add_argument("--out-dir", type=Path, default=HERE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows = logical_rows()
    models = physical_models(rows)
    counts: dict[str, int] = {}
    for row in models:
        counts[str(row["variant"])] = counts.get(str(row["variant"]), 0) + 1
    print(f"plan: {len(rows)} logical rows -> {len(models)} physical encoders")
    print("physical encoders by variant: " + ", ".join(
        f"{name}={counts.get(name, 0)}" for name in DEFAULT_STATE_ROOTS
    ))
    for row in rows:
        print(
            f"  {row['logical_id']:<22} step={int(row['checkpoint_step']):>5} "
            f"{row['model_key']}"
        )
    if args.dry_run:
        return 0

    roots = {
        name: getattr(args, f"{name}_state_root") for name in DEFAULT_STATE_ROOTS
    }
    missing: list[str] = []
    resolved: dict[str, Path] = {}
    for model in models:
        variant = str(model["variant"])
        root = roots[variant]
        checkpoint = resolve_checkpoint(
            root,
            str(model["model_prefix"]),
            int(model["checkpoint_step"]),
        )
        if checkpoint is None:
            missing.append(
                f"{model['model_key']}: state_dict_{model['checkpoint_step']}.ckpt "
                f"under {root}"
            )
        else:
            resolved[str(model["model_key"])] = checkpoint

    if missing:
        print(f"ERROR: {len(missing)} checkpoint(s) missing:", file=sys.stderr)
        for item in missing:
            print(f"  {item}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_list = args.out_dir / "model_list.txt"
    with model_list.open("w", encoding="utf-8") as handle:
        for model in models:
            key = str(model["model_key"])
            handle.write(f"{key} {resolved[key]}\n")

    row_map = args.out_dir / "row_map.csv"
    fields = list(rows[0])
    with row_map.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {model_list} ({len(models)} encoders)")
    print(f"wrote {row_map} ({len(rows)} logical rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
