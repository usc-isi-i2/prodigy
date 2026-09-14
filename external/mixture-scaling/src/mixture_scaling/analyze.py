from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import yaml


METRICS = ("roc_auc_ovr_macro", "f1_macro", "accuracy")
STEPS = (100, 300, 900, 2500)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def run_id(row: dict[str, str]) -> str:
    return Path(row["checkpoint"]).parents[1].name


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def exact_index(
    rows: list[dict[str, str]], key_fields: tuple[str, ...]
) -> dict[tuple[object, ...], dict[str, str]]:
    result: dict[tuple[object, ...], dict[str, str]] = {}
    for row in rows:
        enriched = {**row, "run_id": run_id(row)}
        key = tuple(enriched[field] for field in key_fields)
        if key in result:
            raise ValueError(f"duplicate key {key}")
        result[key] = enriched
    return result


def score_fields(row: dict[str, str]) -> dict[str, float]:
    return {metric: float(row[metric]) for metric in METRICS}


def validate_aggregate(
    rows: list[dict[str, str]], manifest: list[dict[str, str]], name: str
) -> None:
    expected = {
        (entry["run_id"], entry["target"], str(step)): entry
        for entry in manifest
        for step in STEPS
    }
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        rid = run_id(row)
        key = (rid, row["target"], row["checkpoint_step"])
        if key in seen:
            raise ValueError(f"{name}: duplicate result {key}")
        seen.add(key)
        if key not in expected:
            raise ValueError(f"{name}: unexpected result {key}")
        entry = expected[key]
        if row["sources"] != entry["sources"]:
            raise ValueError(f"{name}: source mismatch for {key}")
        if int(row["seed"]) != int(entry["seed"]):
            raise ValueError(f"{name}: seed mismatch for {key}")
        if int(row["support_nodes"]) != 10 * int(row["classes"]):
            raise ValueError(f"{name}: support-label mismatch for {key}")
        for metric, value in score_fields(row).items():
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name}: invalid {metric}={value} for {key}")
    missing = set(expected) - seen
    if missing:
        raise ValueError(f"{name}: missing {len(missing)} results; first={sorted(missing)[0]}")


def log_step_aulc(values: dict[int, float]) -> float:
    if set(values) != set(STEPS):
        raise ValueError(f"expected steps {STEPS}, got {sorted(values)}")
    xs = [math.log10(step) for step in STEPS]
    area = sum(
        (xs[i + 1] - xs[i]) * (values[STEPS[i]] + values[STEPS[i + 1]]) / 2
        for i in range(len(STEPS) - 1)
    )
    return area / (xs[-1] - xs[0])


def build_heldout_ladder(
    graph_order: list[str],
    primary: list[dict[str, str]],
    matrix: list[dict[str, str]],
    ladder: list[dict[str, str]],
) -> list[dict[str, object]]:
    """Reconstruct held-out k=1..6 prefixes without target leakage.

    k=1 is the first non-target single-source model evaluated through the matrix,
    k=2..5 are the dedicated ladder runs, and k=6 is the leave-one-out model.
    """
    primary_i = exact_index(primary, ("run_id", "target", "checkpoint_step"))
    matrix_i = exact_index(matrix, ("run_id", "target", "checkpoint_step"))
    ladder_i = exact_index(ladder, ("target", "sources", "checkpoint_step"))
    output: list[dict[str, object]] = []
    for target in graph_order:
        ordered_sources = [graph for graph in graph_order if graph != target]
        for size in range(1, len(ordered_sources) + 1):
            sources = ordered_sources[:size]
            source_text = ",".join(sources)
            for step in STEPS:
                if size == 1:
                    rid = f"specialist_{sources[0]}_s0"
                    row = matrix_i[(rid, target, str(step))]
                elif size == len(ordered_sources):
                    rid = f"loo_{target}_s0"
                    row = primary_i[(rid, target, str(step))]
                else:
                    row = ladder_i[(target, source_text, str(step))]
                    rid = row["run_id"]
                if row["sources"] != source_text:
                    raise ValueError(
                        f"{target} k={size} step={step}: {row['sources']} != {source_text}"
                    )
                output.append(
                    {
                        "target": target,
                        "mixture_size": size,
                        "sources": source_text,
                        "run_id": rid,
                        "seed": 0,
                        "checkpoint_step": step,
                        **score_fields(row),
                    }
                )
    expected = len(graph_order) * (len(graph_order) - 1) * len(STEPS)
    if len(output) != expected:
        raise ValueError(f"held-out ladder has {len(output)} rows; expected {expected}")
    return output


def summarize_adaptation(ladder: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    grouped: dict[tuple[str, int, str, str], dict[int, float]] = defaultdict(dict)
    for row in ladder:
        for metric in METRICS:
            key = (str(row["target"]), int(row["mixture_size"]), str(row["sources"]), metric)
            grouped[key][int(row["checkpoint_step"])] = float(row[metric])
    per_target: list[dict[str, object]] = []
    for (target, size, sources, metric), values in sorted(grouped.items()):
        best_step = max(STEPS, key=lambda step: values[step])
        per_target.append(
            {
                "target": target,
                "mixture_size": size,
                "sources": sources,
                "metric": metric,
                **{f"step_{step}": values[step] for step in STEPS},
                "log_step_aulc": log_step_aulc(values),
                "best_score": values[best_step],
                "best_step": best_step,
            }
        )
    by_size: list[dict[str, object]] = []
    for metric in METRICS:
        for size in range(1, 7):
            rows = [r for r in per_target if r["metric"] == metric and r["mixture_size"] == size]
            by_size.append(
                {
                    "mixture_size": size,
                    "metric": metric,
                    "targets": len(rows),
                    **{
                        f"mean_step_{step}": statistics.fmean(float(r[f"step_{step}"]) for r in rows)
                        for step in STEPS
                    },
                    "mean_log_step_aulc": statistics.fmean(float(r["log_step_aulc"]) for r in rows),
                }
            )
    return per_target, by_size


def primary_seed_tables(
    primary_s0: list[dict[str, str]], primary_s1_s2: list[dict[str, str]]
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    combined: list[dict[str, object]] = []
    for row in primary_s0 + primary_s1_s2:
        rid = run_id(row)
        family = "leave_one_out" if rid.startswith("loo_") else "specialist"
        combined.append({"run_id": rid, "family": family, **row})
    expected = 7 * 2 * 3 * len(STEPS)
    if len(combined) != expected:
        raise ValueError(f"primary seed table has {len(combined)} rows; expected {expected}")

    summary: list[dict[str, object]] = []
    groups: dict[tuple[str, str, int, str], list[float]] = defaultdict(list)
    for row in combined:
        for metric in METRICS:
            groups[(str(row["target"]), str(row["family"]), int(row["checkpoint_step"]), metric)].append(float(row[metric]))
    for (target, family, step, metric), values in sorted(groups.items()):
        summary.append(
            {
                "target": target,
                "family": family,
                "checkpoint_step": step,
                "metric": metric,
                "seeds": len(values),
                "mean": statistics.fmean(values),
                "sample_std": statistics.stdev(values),
            }
        )

    paired: dict[tuple[str, int, int, str], dict[str, float]] = defaultdict(dict)
    for row in combined:
        for metric in METRICS:
            paired[(str(row["target"]), int(row["seed"]), int(row["checkpoint_step"]), metric)][str(row["family"])] = float(row[metric])
    deltas: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    for (target, _seed, step, metric), values in paired.items():
        if set(values) != {"specialist", "leave_one_out"}:
            raise ValueError(f"incomplete primary pair: {(target, step, metric)}")
        deltas[(target, step, metric)].append(values["leave_one_out"] - values["specialist"])
    contrasts: list[dict[str, object]] = []
    for (target, step, metric), values in sorted(deltas.items()):
        contrasts.append(
            {
                "target": target,
                "checkpoint_step": step,
                "metric": metric,
                "seeds": len(values),
                "mean_loo_minus_specialist": statistics.fmean(values),
                "sample_std": statistics.stdev(values),
                "loo_wins": sum(value > 0 for value in values),
            }
        )
    return combined, summary, contrasts


def matrix_table(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    result = []
    for row in rows:
        if int(row["checkpoint_step"]) != 2500:
            continue
        rid = run_id(row)
        result.append(
            {
                "run_id": rid,
                "family": "leave_one_out" if rid.startswith("loo_") else "specialist",
                "sources": row["sources"],
                "target": row["target"],
                **score_fields(row),
            }
        )
    if len(result) != 14 * 7:
        raise ValueError(f"step-2500 matrix has {len(result)} rows; expected 98")
    return sorted(result, key=lambda row: (str(row["run_id"]), str(row["target"])))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    inputs = {
        "primary_s0": root / "results/primary_s0/primary_results.csv",
        "primary_s1_s2": root / "results/primary_s1_s2/primary_results.csv",
        "matrix_s0": root / "results/matrix_s0/matrix_results.csv",
        "ladder_s0": root / "results/ladder_s0/ladder_results.csv",
    }
    source_rows = {name: read_csv(path) for name, path in inputs.items()}
    manifests = {
        "primary_s0": read_tsv(root / "manifests/primary.tsv"),
        "primary_s1_s2": read_tsv(root / "manifests/primary_s1_s2.tsv"),
        "matrix_s0": read_tsv(root / "manifests/matrix_s0_eval.tsv"),
        "ladder_s0": read_tsv(root / "manifests/ladder_s0_eval.tsv"),
    }
    for name, rows in source_rows.items():
        validate_aggregate(rows, manifests[name], name)
    config = yaml.safe_load((root / "configs/graphs.yaml").read_text())
    graph_order = list(config["graphs"])

    heldout = build_heldout_ladder(
        graph_order,
        source_rows["primary_s0"],
        source_rows["matrix_s0"],
        source_rows["ladder_s0"],
    )
    adaptation, adaptation_by_size = summarize_adaptation(heldout)
    primary, primary_summary, primary_contrasts = primary_seed_tables(
        source_rows["primary_s0"], source_rows["primary_s1_s2"]
    )
    matrix = matrix_table(source_rows["matrix_s0"])

    output_root = root / "results/analysis"
    outputs = {
        "heldout_ladder.csv": heldout,
        "adaptation_by_target.csv": adaptation,
        "adaptation_by_size.csv": adaptation_by_size,
        "primary_all_seeds.csv": primary,
        "primary_summary.csv": primary_summary,
        "primary_contrasts.csv": primary_contrasts,
        "matrix_step2500.csv": matrix,
    }
    for name, rows in outputs.items():
        write_csv(output_root / name, rows)
    audit = {
        "inputs": {
            name: {"path": str(path.relative_to(root)), "rows": len(source_rows[name]), "sha256": sha256(path)}
            for name, path in inputs.items()
        },
        "outputs": {name: len(rows) for name, rows in outputs.items()},
        "invariants": {
            "graphs": graph_order,
            "steps": list(STEPS),
            "labels_per_class": 10,
            "aggregate_exact_manifest_coverage": True,
            "aggregate_metrics_finite_and_bounded": True,
            "heldout_ladder_target_leakage": False,
            "adaptation_efficiency": "normalized trapezoidal area under score vs log10(checkpoint_step)",
        },
    }
    (output_root / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
