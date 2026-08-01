#!/usr/bin/env python3
"""Resolve the eight single-source models to pinned matched-40k checkpoints."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

MODELS = [
    ("ukr_rus_twitter", "nm_ss_ukr_rus_twitter", "original NM matrix"),
    ("covid19_twitter", "nm_ss_covid19_twitter", "original NM matrix"),
    ("midterm", "nm_ss_midterm", "original NM matrix"),
    ("covid_political", "nm_ss_covid_political", "original NM matrix"),
    ("election2020", "nm_ss_election2020", "original NM matrix"),
    ("ukr_rus_suspended", "nm_ss_ukr_rus_suspended", "original NM matrix"),
    ("twibot20", "nm_ss_twibot20", "original NM matrix"),
    ("cp_hk_twitter", "nm_ss_cp_hk_twitter", "original NM matrix"),
]


def resolve(state_dir: Path, prefix: str, step: int) -> Path | None:
    runs = sorted(
        (path for path in state_dir.glob(f"{prefix}_*") if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for run in runs:
        checkpoint = run / "checkpoint" / f"state_dict_{step}.ckpt"
        if checkpoint.is_file():
            return checkpoint.resolve()
    return None


def main() -> int:
    here = Path(__file__).resolve().parent
    repo = here.parents[3]
    analysis_data = repo / "scripts/experiments/analysis/nm_single_source_downstream/data"
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-state-dir", default="/dataMeR1/phil/gfm/prodigy/state")
    parser.add_argument("--step", type=int, default=40000)
    parser.add_argument("--model-list", default=str(here / "model_list.txt"))
    parser.add_argument("--manifest", default=str(analysis_data / "model_manifest.csv"))
    args = parser.parse_args()

    resolved = []
    missing = []
    for source, model, provenance in MODELS:
        state_dir = Path(args.reuse_state_dir)
        checkpoint = resolve(state_dir, model, args.step)
        if checkpoint is None:
            missing.append(f"{model} under {state_dir}")
        else:
            resolved.append((source, model, provenance, args.step, checkpoint))

    if missing:
        print(f"ERROR: missing {len(missing)} checkpoint(s):")
        for item in missing:
            print(f"  {item}")
        return 1

    model_list = Path(args.model_list)
    model_list.parent.mkdir(parents=True, exist_ok=True)
    with model_list.open("w", encoding="utf-8") as handle:
        for _, model, _, _, checkpoint in resolved:
            handle.write(f"{model} {checkpoint}\n")

    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["source", "model", "provenance", "checkpoint_step", "checkpoint"])
        writer.writerows(resolved)

    print(f"wrote {model_list} ({len(resolved)} models)")
    print(f"wrote {manifest}")
    for source, model, provenance, step, checkpoint in resolved:
        print(f"  {source:22s} {model:28s} step={step} {provenance}: {checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
