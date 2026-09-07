#!/usr/bin/env python3
"""Derive a preregistered label-free weighted source schedule from TRACE health."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from itertools import groupby
from pathlib import Path

import torch

from scripts.experiments.setup.trace_health_guided.make_stage1_plan import SOURCES
from scripts.experiments.setup.trace_schedule_scaling.export_replays import (
    support_loo_prototype_accuracy,
)


RANK_WEIGHTS = (4, 3, 2, 1)
SUPPORT_COMPETENCE_THRESHOLD = 0.55


def allocate_rank_counts(total_steps: int) -> tuple[int, ...]:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    denominator = sum(RANK_WEIGHTS)
    exact = [total_steps * weight / denominator for weight in RANK_WEIGHTS]
    counts = [int(value) for value in exact]
    remainder = total_steps - sum(counts)
    order = sorted(
        range(len(counts)),
        key=lambda index: (-(exact[index] - counts[index]), index),
    )
    for index in order[:remainder]:
        counts[index] += 1
    return tuple(counts)


def weighted_fair_sequence(
    ranked_sources: tuple[str, ...], counts: tuple[int, ...]
) -> tuple[str, ...]:
    if len(ranked_sources) != len(counts) or sum(counts) <= 0:
        raise ValueError("ranked_sources and positive counts must align")
    total = sum(counts)
    used = Counter()
    result = []
    for step in range(1, total + 1):
        eligible = [
            (index, source)
            for index, source in enumerate(ranked_sources)
            if used[source] < counts[index]
        ]
        index, source = max(
            eligible,
            key=lambda item: (
                step * counts[item[0]] / total - used[item[1]],
                -item[0],
            ),
        )
        result.append(source)
        used[source] += 1
    if tuple(used[source] for source in ranked_sources) != counts:
        raise AssertionError("weighted-fair construction changed source counts")
    return tuple(result)


def compress(sequence: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[int, ...]]:
    grouped = [(source, len(tuple(values))) for source, values in groupby(sequence)]
    return tuple(source for source, _ in grouped), tuple(count for _, count in grouped)


def read_specialists(model_list: Path) -> dict[str, tuple[str, int]]:
    result = {}
    with model_list.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            sources = tuple(value for value in row["sources"].split(",") if value)
            if len(sources) != 1:
                continue
            model_id = row["model_id"]
            try:
                seed = int(model_id.rsplit("_s", 1)[1])
            except (IndexError, ValueError) as error:
                raise ValueError(f"cannot parse specialist seed: {model_id}") from error
            result[model_id] = (sources[0], seed)
    expected = len(SOURCES) * 3
    if len(result) != expected:
        raise ValueError(f"expected {expected} specialists, got {len(result)}")
    return result


def full_rows(directory: Path, model_ids: set[str]) -> dict[str, dict]:
    rows = [
        json.loads(line)
        for line in (directory / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    selected = {
        row["model_id"]: row
        for row in rows
        if row["variant"] == "baseline" and row["decoder"] == "full_model"
    }
    if set(selected) != model_ids:
        raise ValueError("replay model set does not match specialist model list")
    return selected


def measure_health(
    directory: Path, specialists: dict[str, tuple[str, int]]
) -> dict[str, dict[str, float]]:
    full_rows(directory, set(specialists))
    batches = [
        torch.load(
            directory / "batches" / f"batch_{index:03d}.pt",
            map_location="cpu",
            weights_only=False,
        )
        for index in range(32)
    ]
    output = {}
    for model_id in sorted(specialists):
        records = torch.load(
            directory / f"{model_id}__baseline.pt",
            map_location="cpu",
            weights_only=False,
        )
        if len(records) != len(batches):
            raise ValueError(f"incomplete replay records for {model_id}")
        agreements = []
        loo = []
        for record, batch in zip(records, batches):
            full = record["logits"]["full_model"].argmax(1)
            u1 = record["logits"]["U1_pre_meta/ridge"].argmax(1)
            agreements.append((full == u1).float())
            loo.append(
                support_loo_prototype_accuracy(
                    record["embeddings"]["U1_pre_meta"], batch
                )
            )
        output[model_id] = {
            "u1_agreement": float(torch.cat(agreements).mean()),
            "support_loo": float(torch.cat(loo).mean()),
        }
    return output


def build_rows(
    specialists: dict[str, tuple[str, int]],
    health: dict[str, dict[str, float]],
    total_steps: int,
    competence_threshold: float = SUPPORT_COMPETENCE_THRESHOLD,
) -> list[dict[str, str]]:
    rank_counts = allocate_rank_counts(total_steps)
    rows = []
    for seed in sorted({seed for _, seed in specialists.values()}):
        by_source = {
            source: health[model_id]
            for model_id, (source, model_seed) in specialists.items()
            if model_seed == seed
        }
        if set(by_source) != set(SOURCES):
            raise ValueError(f"seed {seed} does not contain all sources")
        target_competence = max(values["support_loo"] for values in by_source.values())
        if target_competence < competence_threshold:
            raise ValueError(
                f"seed {seed} failed support-competence gate: "
                f"{target_competence:.6f} < {competence_threshold:.6f}"
            )
        ranked = tuple(
            sorted(
                SOURCES,
                key=lambda source: (
                    -by_source[source]["u1_agreement"],
                    -by_source[source]["support_loo"],
                    SOURCES.index(source),
                ),
            )
        )
        sequence = weighted_fair_sequence(ranked, rank_counts)
        segment_sources, segment_steps = compress(sequence)
        rows.append(
            {
                "model_id": f"health_guided_s{seed}",
                "condition": "health_guided",
                "seed": str(seed),
                "sources": ",".join(SOURCES),
                "ranked_sources": ",".join(ranked),
                "u1_agreement_by_source": ";".join(
                    f"{source}={by_source[source]['u1_agreement']:.9f}"
                    for source in SOURCES
                ),
                "support_loo_by_source": ";".join(
                    f"{source}={by_source[source]['support_loo']:.9f}"
                    for source in SOURCES
                ),
                "target_support_competence": f"{target_competence:.9f}",
                "rank_counts": ",".join(str(value) for value in rank_counts),
                "segment_sources": ",".join(segment_sources),
                "segment_steps": ",".join(str(value) for value in segment_steps),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-directory", type=Path, required=True)
    parser.add_argument("--model-list", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--total-steps", type=int, default=2500)
    parser.add_argument(
        "--support-competence-threshold",
        type=float,
        default=SUPPORT_COMPETENCE_THRESHOLD,
    )
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"refusing to overwrite health plan: {args.output}")
    specialists = read_specialists(args.model_list)
    health = measure_health(args.replay_directory, specialists)
    rows = build_rows(
        specialists,
        health,
        args.total_steps,
        args.support_competence_threshold,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"OK: wrote {len(rows)} health-guided schedules to {args.output}")
    for row in rows:
        print(
            f"{row['model_id']}: {row['ranked_sources']} "
            f"counts={row['rank_counts']} competence={row['target_support_competence']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
