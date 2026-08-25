#!/usr/bin/env python3
"""Assemble compute-matched n_hop=2 saturation and compare it with n_hop=1."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


HERE = Path(__file__).resolve().parent
REPO_ROOT = next(p for p in HERE.parents if (p / "AGENTS.md").is_file())
H1_ROOT = REPO_ROOT / "scripts/experiments/analysis/transfer/ablations/prodigy_nm/saturation/pretrain_saturation"
ARMS = ("all8", "ukr", "covid")
STEPS = (0, 100, 500, 1_000, 2_000, 10_000, 40_000)
CLASSIFICATION_DATASETS = (
    "covid_political",
    "election2020",
    "ukr_rus_suspended",
    "twibot20",
)
REGRESSION_DATASETS = (
    "covid19_twitter",
    "midterm",
    "twibot20",
    "ukr_rus_twitter",
)
TARGETS = ("followers_count", "account_age_days")
ARM_COLOR = {"all8": "#2a78d6", "ukr": "#eb6834", "covid": "#1baf7a"}
MODEL_RE = re.compile(r"^sat_h2m_(?P<arm>all8|ukr|covid)_s(?P<step>\d{6})$")
SHARED_STEP0 = "sat_h2m_shared_s000000"
RUN_RE = re.compile(
    r"^eval_(?P<model>sat_h2m_(?:shared|all8|ukr|covid)_s\d{6})_to_"
    r"(?P<dataset>covid_political|election2020|ukr_rus_suspended|twibot20)_"
    r"pl_10shot(?:_|$)"
)


def parse_h2_model(model: str) -> tuple[str, int]:
    if model == SHARED_STEP0:
        return "shared", 0
    match = MODEL_RE.match(model)
    if match is None:
        raise ValueError(f"not an n_hop=2 saturation key: {model!r}")
    arm, step = match["arm"], int(match["step"])
    if step not in STEPS or step == 0:
        raise ValueError(f"unregistered arm checkpoint: {model!r}")
    return arm, step


def expected_raw_models() -> set[str]:
    return {SHARED_STEP0} | {
        f"sat_h2m_{arm}_s{step:06d}"
        for arm in ARMS
        for step in STEPS
        if step > 0
    }


def read_test_auc(run_dir: Path) -> tuple[float, Path] | None:
    metrics_paths = sorted((run_dir / "data").glob("metrics_test*.json"), reverse=True)
    for path in metrics_paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return float(payload["test_roc_auc"]), path
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return None


def collect_classification(log_root: Path) -> pd.DataFrame:
    newest: dict[tuple[str, str], tuple[float, float, Path]] = {}
    if not log_root.is_dir():
        raise FileNotFoundError(f"log root not found: {log_root}")
    for run_dir in log_root.glob("eval_sat_h2m_*_to_*_pl_10shot*"):
        if not run_dir.is_dir():
            continue
        match = RUN_RE.match(run_dir.name)
        if match is None:
            continue
        result = read_test_auc(run_dir)
        if result is None:
            continue
        value, metrics_path = result
        key = (match["model"], match["dataset"])
        candidate = (run_dir.stat().st_mtime, value, metrics_path)
        if key not in newest or candidate[0] > newest[key][0]:
            newest[key] = candidate

    expected = {
        (model, dataset)
        for model in expected_raw_models()
        for dataset in CLASSIFICATION_DATASETS
    }
    missing = sorted(expected - newest.keys())
    if missing:
        preview = ", ".join(f"{model}/{dataset}" for model, dataset in missing[:8])
        raise ValueError(f"classification matrix incomplete: {len(missing)} missing ({preview})")

    return pd.DataFrame([
        {
            "model": model,
            "dataset": dataset,
            "target": "",
            "value": newest[(model, dataset)][1],
            "evidence_path": str(newest[(model, dataset)][2]),
        }
        for model, dataset in sorted(expected)
    ])


def collect_regression(probe_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for dataset in REGRESSION_DATASETS:
        path = probe_dir / f"{dataset}__reg_probe.csv"
        if not path.is_file():
            raise ValueError(f"missing regression probe output: {path}")
        frame = pd.read_csv(path)
        frame["evidence_path"] = str(path)
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    raw = raw[(raw["target"].isin(TARGETS)) & (raw["alpha"] == 1.0)].copy()
    provenance = {"n_hop": 2, "hop_sizes": "9,9", "node_limit": 101}
    for column, expected_value in provenance.items():
        if column not in raw.columns:
            raise ValueError(f"regression output lacks matched-sampler column {column!r}")
        actual_values = set(raw[column].dropna().tolist())
        if actual_values != {expected_value}:
            raise ValueError(
                f"regression output has {column}={actual_values}, expected {expected_value!r}"
            )

    encoder = raw[raw["model"].isin(expected_raw_models())].copy()
    expected = {
        (model, dataset, target)
        for model in expected_raw_models()
        for dataset in REGRESSION_DATASETS
        for target in TARGETS
    }
    actual = set(zip(encoder.model, encoder.dataset, encoder.target))
    missing = sorted(expected - actual)
    duplicates = encoder.duplicated(["model", "dataset", "target"], keep=False)
    if missing:
        raise ValueError(f"regression matrix incomplete: {len(missing)} missing")
    if duplicates.any():
        raise ValueError("regression matrix has duplicate model/dataset/target rows")

    floors = raw[raw.model == "__features_only__"].copy()
    if set(zip(floors.dataset, floors.target)) != {
        (dataset, target) for dataset in REGRESSION_DATASETS for target in TARGETS
    }:
        raise ValueError("regression raw-feature floors are incomplete or duplicated")
    encoder["value"] = encoder["spearman"].astype(float)
    return encoder[["model", "dataset", "target", "value", "evidence_path"]], floors


def expand_shared_step0(raw: pd.DataFrame, task: str, metric: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for record in raw.to_dict("records"):
        source_model = str(record["model"])
        arm, step = parse_h2_model(source_model)
        row_arms = ARMS if arm == "shared" else (arm,)
        for row_arm in row_arms:
            rows.append({
                "arm": row_arm,
                "step": step,
                "task": task,
                "dataset": record["dataset"],
                "target": record.get("target", "") or "",
                "metric": metric,
                "value": float(record["value"]),
                "model": f"sat_h2m_{row_arm}_s{step:06d}",
                "source_model": source_model,
                "shared_step0": source_model == SHARED_STEP0,
                "n_hop": 2,
                "hop_sizes": "9,9",
                "node_limit": 101,
                "nm_walk_hops": 1,
                "evidence_path": record.get("evidence_path", ""),
            })
    return pd.DataFrame(rows)


def load_h1() -> pd.DataFrame:
    original = pd.read_csv(H1_ROOT / "data" / "pretrain_saturation_long.csv")
    classification = original[original.task == "classification"].copy()
    classification["target"] = ""

    probe_frames = []
    for dataset in REGRESSION_DATASETS:
        path = H1_ROOT / "data" / "reg_probe" / f"{dataset}__reg_probe.csv"
        frame = pd.read_csv(path)
        frame = frame[
            frame.model.astype(str).str.startswith("sat_") & frame.target.isin(TARGETS)
        ].copy()
        key = frame.model.str.extract(r"^sat_(?P<arm>all8|ukr|covid)_s(?P<step>\d+)$")
        frame["arm"] = key.arm
        frame["step"] = key.step.astype(int)
        frame["task"] = "regression"
        frame["metric"] = "spearman"
        frame["value"] = frame.spearman
        probe_frames.append(frame)
    regression = pd.concat(probe_frames, ignore_index=True)

    base = pd.concat([
        classification[["arm", "step", "task", "dataset", "target", "metric", "value"]],
        regression[["arm", "step", "task", "dataset", "target", "metric", "value"]],
    ], ignore_index=True)

    step0 = pd.read_csv(H1_ROOT / "data" / "step0_anchor.csv")
    step0["target"] = step0.target.fillna("")
    step0_rows = []
    for record in step0.to_dict("records"):
        for arm in ARMS:
            step0_rows.append({
                "arm": arm,
                "step": 0,
                "task": record["task"],
                "dataset": record["dataset"],
                "target": record["target"],
                "metric": record["metric"],
                "value": record["value"],
            })
    return pd.concat([base, pd.DataFrame(step0_rows)], ignore_index=True)


def summarize(h2: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (arm, task, target), group in h2.groupby(["arm", "task", "target"], dropna=False):
        means = group.groupby("step").value.mean().sort_index()
        row = {
            "arm": arm,
            "task": task,
            "target": target,
            "step0": means.get(0, float("nan")),
            "step100": means.get(100, float("nan")),
            "step500": means.get(500, float("nan")),
            "step40000": means.get(40_000, float("nan")),
            "best": means.max(),
            "best_step": int(means.idxmax()),
            "plateau_spread_500_40000": means[means.index >= 500].max()
            - means[means.index >= 500].min(),
        }
        denominator = row["best"] - row["step100"]
        row["gain_fraction_by_500"] = (
            (row["step500"] - row["step100"]) / denominator
            if denominator > 0 else float("nan")
        )
        rows.append(row)
    return pd.DataFrame(rows)


def make_figure(h1: pd.DataFrame, h2: pd.DataFrame, out: Path) -> None:
    panels = [
        ("classification", "", "ROC-AUC", "Classification · mean over 4 graphs"),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    axes = (ax,)
    for ax, (task, target, ylabel, title) in zip(axes, panels):
        for radius, frame, style in ((1, h1, "--"), (2, h2, "-")):
            selected = frame[(frame.task == task) & (frame.target.fillna("") == target)]
            for arm in ARMS:
                series = selected[selected.arm == arm].groupby("step").value.mean().sort_index()
                ax.plot(
                    series.index,
                    series.values,
                    linestyle=style,
                    color=ARM_COLOR[arm],
                    linewidth=2 if radius == 2 else 1.4,
                    marker="o" if radius == 2 else None,
                    markersize=4.5,
                    label=f"{arm} · h{radius}",
                )
        ax.set_xscale("linear")
        ax.set_xlim(0, max(STEPS))
        ax.set_xticks((0, 10_000, 20_000, 30_000, 40_000))
        ax.set_xticklabels(("0", "10k", "20k", "30k", "40k"))
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlabel("pretraining steps")
        ax.set_ylabel(ylabel)
        ax.grid(True, color="#e4e3dd", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        fontsize=8,
    )
    fig.suptitle("Classification saturation: compute-matched two-hop context vs one hop",
                 y=0.995, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.78])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--probe-dir", type=Path, default=HERE / "data" / "reg_probe")
    parser.add_argument("--out-dir", type=Path, default=HERE / "data")
    args = parser.parse_args()

    classification = expand_shared_step0(
        collect_classification(args.log_root), "classification", "roc_auc"
    )
    regression_raw, floors = collect_regression(args.probe_dir)
    regression = expand_shared_step0(regression_raw, "regression", "spearman")
    h2 = pd.concat([classification, regression], ignore_index=True).sort_values(
        ["task", "target", "arm", "step", "dataset"]
    )
    expected = len(ARMS) * len(STEPS) * (
        len(CLASSIFICATION_DATASETS) + len(REGRESSION_DATASETS) * len(TARGETS)
    )
    if len(h2) != expected:
        raise ValueError(f"expanded h2 table has {len(h2)} rows, expected {expected}")

    h1 = load_h1()
    key = ["arm", "step", "task", "dataset", "target", "metric"]
    comparison = h2.merge(h1, on=key, how="left", suffixes=("_h2", "_h1"), validate="one_to_one")
    if comparison.value_h1.isna().any():
        raise ValueError("n_hop=1 baseline is missing paired cells")
    comparison["delta_h2_minus_h1"] = comparison.value_h2 - comparison.value_h1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    h2.to_csv(args.out_dir / "pretrain_saturation_nhop2_long.csv", index=False)
    comparison.to_csv(args.out_dir / "nhop_comparison.csv", index=False)
    summarize(h2).to_csv(args.out_dir / "summary.csv", index=False)
    floors.to_csv(args.out_dir / "regression_floors.csv", index=False)
    make_figure(h1, h2, HERE / "figures" / "nhop_comparison.png")
    print(f"wrote complete h2 table ({len(h2)} rows) and paired comparison")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
