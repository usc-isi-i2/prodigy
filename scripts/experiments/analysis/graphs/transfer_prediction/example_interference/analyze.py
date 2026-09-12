"""Aggregate per-query rescue/break interference from matched replay exports.

The pair model is treated as an intervention on each constituent singleton:
``singleton A -> pair A+B`` and ``singleton B -> pair A+B``.  Raw target inputs
and evaluation episodes must be identical within a target and stream.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch


STAGES = [
    "raw_joint/ridge",
    "S0_conv_center/ridge",
    "S0_pool/ridge",
    "U1_pre_meta/ridge",
    "M2_post_meta/ridge",
    "final_input/ridge",
    "full_model",
]


def read_metrics(root: Path) -> dict[tuple[str, str], dict]:
    rows = {}
    for path in sorted(root.glob("*/metrics.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row.get("variant") != "baseline" or row.get("decoder") not in STAGES:
                continue
            key = (row["dataset"], row["model_id"], row["decoder"])
            if key in rows:
                raise ValueError(f"duplicate metric cell: {key}")
            rows[key] = row
    return rows


def expected_labels(n: int, n_query: int = 4, n_way: int = 2) -> torch.Tensor:
    per_episode = torch.arange(n_way).repeat_interleave(n_query)
    if n % len(per_episode):
        raise ValueError(f"query count {n} is not divisible by {len(per_episode)}")
    return per_episode.repeat(n // len(per_episode))


def load_logits(path: Path) -> dict[str, torch.Tensor]:
    # These are trusted experiment exports created by replay.py. weights_only
    # prevents arbitrary pickle globals even though the files are internal.
    records = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(records, list) or not records:
        raise ValueError(f"invalid replay export: {path}")
    batch_ids = [int(r["batch"]) for r in records]
    if batch_ids != list(range(len(records))):
        raise ValueError(f"non-contiguous batches: {path}")
    if len({r["batch_sha256"] for r in records}) != len(records):
        raise ValueError(f"duplicate batch hashes: {path}")
    decoders = set(records[0]["logits"])
    if not set(STAGES).issubset(decoders):
        raise ValueError(f"missing required stages in {path}")
    if any(set(r["logits"]) != decoders for r in records):
        raise ValueError(f"decoder set changes across batches: {path}")
    return {stage: torch.cat([r["logits"][stage].float() for r in records]) for stage in STAGES}


def sources_from_pair(model_id: str, known_sources: set[str]) -> tuple[str, str]:
    if not model_id.startswith("nmpair_"):
        raise ValueError(model_id)
    body = model_id.removeprefix("nmpair_")
    matches = [(a, b) for a in known_sources for b in known_sources if a < b and body in {f"{a}__{b}", f"{b}__{a}"}]
    if len(matches) != 1:
        raise ValueError(f"cannot uniquely parse pair model {model_id}: {matches}")
    return matches[0]


def outcome(base: torch.Tensor, pair: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
    if base.shape != pair.shape or base.ndim != 2 or base.shape[1] != 2:
        raise ValueError(f"expected matching binary logits, got {base.shape} and {pair.shape}")
    if len(labels) != len(base):
        raise ValueError("label/logit length mismatch")
    bpred, ppred = base.argmax(1), pair.argmax(1)
    bc, pc = bpred.eq(labels), ppred.eq(labels)
    idx = torch.arange(len(labels))
    other = 1 - labels
    bm = base[idx, labels] - base[idx, other]
    pm = pair[idx, labels] - pair[idx, other]
    return {
        "base_correct": bc,
        "pair_correct": pc,
        "rescued": ~bc & pc,
        "broken": bc & ~pc,
        "both_correct": bc & pc,
        "both_wrong": ~bc & ~pc,
        "margin_change": pm - bm,
    }


def summarize(values: dict[str, torch.Tensor]) -> dict[str, float | int]:
    n = len(values["base_correct"])
    counts = {k: int(values[k].sum()) for k in ("rescued", "broken", "both_correct", "both_wrong")}
    if sum(counts.values()) != n:
        raise ValueError("outcome partition failed")
    base_correct = int(values["base_correct"].sum())
    pair_correct = int(values["pair_correct"].sum())
    return {
        "queries": n,
        **counts,
        "base_accuracy": base_correct / n,
        "pair_accuracy": pair_correct / n,
        "net_accuracy_change": (pair_correct - base_correct) / n,
        "churn": (counts["rescued"] + counts["broken"]) / n,
        "rescue_rate_among_base_errors": counts["rescued"] / (n - base_correct) if base_correct < n else None,
        "break_rate_among_base_correct": counts["broken"] / base_correct if base_correct else None,
        "mean_true_margin_change": float(values["margin_change"].mean()),
    }


def verify_accuracy(summary: dict, metric: dict, label: str, atol: float = 1e-8) -> None:
    if abs(float(summary["base_accuracy"]) - float(metric["accuracy"])) > atol:
        raise ValueError(f"{label} reconstructed accuracy does not match metrics")


def analyze(singleton_root: Path, pair_root: Path, stream: str) -> tuple[list[dict], list[dict], dict]:
    singleton_metrics = read_metrics(singleton_root)
    pair_metrics = read_metrics(pair_root)
    known_sources = {model.removeprefix("ss_") for _, model, _ in singleton_metrics if model.startswith("ss_")}
    pair_files = sorted(pair_root.glob("*/nmpair_*__baseline.pt"))
    stage_rows, intervention_rows = [], []
    fingerprints = set()

    for pair_path in pair_files:
        target = pair_path.parent.name
        pair_model = pair_path.name.removesuffix("__baseline.pt")
        source_a, source_b = sources_from_pair(pair_model, known_sources)
        pair_logits = load_logits(pair_path)
        pair_metric = pair_metrics[(target, pair_model, "full_model")]
        fingerprints.add((stream, target, pair_metric["episode_fingerprint"]))
        labels = expected_labels(len(pair_logits["full_model"]))
        pair_full_summary = summarize(outcome(pair_logits["full_model"], pair_logits["full_model"], labels))
        verify_accuracy(pair_full_summary, pair_metric, f"{stream}/{target}/{pair_model}")

        for base_source, added_source in ((source_a, source_b), (source_b, source_a)):
            singleton_model = f"ss_{base_source}"
            singleton_path = singleton_root / target / f"{singleton_model}__baseline.pt"
            if not singleton_path.exists():
                raise ValueError(f"missing singleton replay: {singleton_path}")
            singleton_logits = load_logits(singleton_path)
            base_metric = singleton_metrics[(target, singleton_model, "full_model")]
            if base_metric["episode_fingerprint"] != pair_metric["episode_fingerprint"]:
                raise ValueError(f"episode mismatch for {target}: {singleton_model} vs {pair_model}")

            per_stage = {}
            for order, stage in enumerate(STAGES):
                values = outcome(singleton_logits[stage], pair_logits[stage], labels)
                row = {
                    "stream": stream,
                    "target": target,
                    "base_source": base_source,
                    "added_source": added_source,
                    "pair_model": pair_model,
                    "stage_order": order,
                    "stage": stage,
                    **summarize(values),
                }
                stage_rows.append(row)
                per_stage[stage] = values
            verify_accuracy(stage_rows[-1], base_metric, f"{stream}/{target}/{singleton_model}")

            full = per_stage["full_model"]
            final_flips = full["rescued"] | full["broken"]
            first = Counter()
            for idx in torch.where(final_flips)[0].tolist():
                first_stage = "no_earlier_correctness_divergence"
                for stage in STAGES[:-1]:
                    if per_stage[stage]["base_correct"][idx] != per_stage[stage]["pair_correct"][idx]:
                        first_stage = stage
                        break
                first[first_stage] += 1
            result = {
                "stream": stream,
                "target": target,
                "base_source": base_source,
                "added_source": added_source,
                "pair_model": pair_model,
                **summarize(full),
            }
            for stage in [*STAGES[:-1], "no_earlier_correctness_divergence"]:
                result[f"first_divergence__{stage}"] = first[stage]
            intervention_rows.append(result)

    expected_cells = len(pair_files) * 2
    if len(intervention_rows) != expected_cells or len(stage_rows) != expected_cells * len(STAGES):
        raise ValueError("incomplete output")
    receipt = {
        "protocol": "example_level_source_addition_interference_v1",
        "stream": stream,
        "pair_models": len(pair_files),
        "directional_interventions": len(intervention_rows),
        "stage_rows": len(stage_rows),
        "targets": sorted({r["target"] for r in intervention_rows}),
        "queries_per_intervention": sorted({r["queries"] for r in intervention_rows}),
        "stages": STAGES,
        "episode_fingerprints": [list(x) for x in sorted(fingerprints)],
        "scope": "Saved seed-0 classification replays; fixed queries and episode inputs; no new training or inference; target examples are not independent training replicates.",
    }
    return stage_rows, intervention_rows, receipt


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--singleton-root", type=Path, required=True)
    parser.add_argument("--pair-root", type=Path, required=True)
    parser.add_argument("--stream", choices=("original", "fresh"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    stage_rows, intervention_rows, receipt = analyze(args.singleton_root, args.pair_root, args.stream)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / f"{args.stream}_stage_outcomes.csv", stage_rows)
    write_csv(args.output / f"{args.stream}_interference.csv", intervention_rows)
    for name in (f"{args.stream}_stage_outcomes.csv", f"{args.stream}_interference.csv"):
        receipt[f"{name}_sha256"] = hashlib.sha256((args.output / name).read_bytes()).hexdigest()
    (args.output / f"{args.stream}_validation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
