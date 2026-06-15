#!/usr/bin/env python3
"""Collect local eval metrics into a reproducible CSV, with optional plots.

Preferred usage is with a frozen run-list manifest:

    python scripts/analysis/export_eval_results_csv.py \
      --run-list scripts/analysis/manifests/merged_ukr_rus_covid_nm_eval_runs_v1.txt \
      --out-dir /dataMeR1/phil/gfm/prodigy/log/eval_merged_ukr_rus_covid_nm_csv_v1

The manifest should contain one eval log directory per line. Lines beginning
with "#" are ignored.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import re
from typing import Any

np: Any = None
pd: Any = None
plt: Any = None


DEFAULT_DATASETS = (
    "covid19_twitter",
    "ukr_rus_twitter",
    "covid_political",
    "election2020",
    "ukr_rus_suspended",
    "midterm",
)

TASK_LABELS = {
    "nm": "neighbor_matching",
    "lp": "temporal_link_prediction",
    "pl": "classification",
}


def load_table_dependencies() -> None:
    global pd
    try:
        import pandas as pd_mod
    except ImportError as exc:
        raise SystemExit("Missing pandas. Run this from the conda env with pandas installed.") from exc

    pd = pd_mod


def load_plot_dependencies() -> None:
    global np, plt
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt_mod
        import numpy as np_mod
    except ImportError as exc:
        raise SystemExit(
            "Missing plotting dependency. Install numpy and matplotlib, or omit --plot."
        ) from exc

    np = np_mod
    plt = plt_mod


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def read_run_list(path: Path, log_root: Path | None) -> list[Path]:
    runs: list[Path] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        run_path = Path(line)
        if not run_path.is_absolute() and log_root is not None:
            run_path = log_root / run_path
        runs.append(run_path)
    return runs


def discover_runs(log_root: Path, run_glob: str) -> list[Path]:
    return [Path(path) for path in sorted(glob.glob(str(log_root / run_glob)))]


def checkpoint_label(model_name: str) -> str:
    last = model_name.rsplit("_", 1)[-1]
    if re.fullmatch(r"\d+[kKmM]?", last):
        return last
    return model_name


def parse_run_name(run_name: str, datasets: list[str]) -> dict[str, Any]:
    if not run_name.startswith("eval_") or "_to_" not in run_name:
        raise ValueError(f"Cannot parse eval run name: {run_name}")

    model_name, rest = run_name[len("eval_") :].split("_to_", 1)
    dataset_name = None
    remainder = None
    for candidate in sorted(datasets, key=len, reverse=True):
        prefix = f"{candidate}_"
        if rest.startswith(prefix):
            dataset_name = candidate
            remainder = rest[len(prefix) :]
            break

    if dataset_name is None or remainder is None:
        fallback = re.match(
            r"(?P<dataset>.+)_(?P<task>nm|lp|pl)_(?P<shots>\d+)shot(?:_(?P<timestamp>.*))?$",
            rest,
        )
        if fallback is None:
            raise ValueError(f"Cannot parse dataset/task/shot from run name: {run_name}")
        dataset_name = fallback.group("dataset")
        task = fallback.group("task")
        shots = int(fallback.group("shots"))
        timestamp = fallback.group("timestamp") or ""
    else:
        match = re.match(
            r"(?P<task>nm|lp|pl)_(?P<shots>\d+)shot(?:_(?P<timestamp>.*))?$",
            remainder,
        )
        if match is None:
            raise ValueError(f"Cannot parse task/shot from run name: {run_name}")
        task = match.group("task")
        shots = int(match.group("shots"))
        timestamp = match.group("timestamp") or ""

    return {
        "run_name": run_name,
        "model": model_name,
        "checkpoint": checkpoint_label(model_name),
        "dataset": dataset_name,
        "task": task,
        "task_name": TASK_LABELS.get(task, task),
        "shots": shots,
        "timestamp": timestamp,
    }


def split_from_metrics_path(path: Path) -> str:
    name = path.name
    match = re.match(r"metrics_(?P<split>.+?)(?:_step\d+)?\.json$", name)
    if match is None:
        raise ValueError(f"Cannot parse split from metrics file name: {path}")
    return match.group("split")


def numeric_metrics(payload: dict[str, Any], split: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    prefix = f"{split}_"
    for key, value in payload.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        metric_name = key[len(prefix) :] if key.startswith(prefix) else key
        metrics[metric_name] = float(value)
    return metrics


def collect_rows(
    run_dirs: list[Path],
    datasets: list[str],
    *,
    missing_metrics_policy: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
        run_meta = parse_run_name(run_dir.name, datasets)
        metrics_paths = sorted((run_dir / "data").glob("metrics_*.json"))
        if not metrics_paths:
            msg = f"No metrics_*.json files found under: {run_dir / 'data'}"
            if missing_metrics_policy == "skip":
                print(f"[skip] {msg}")
                continue
            raise FileNotFoundError(msg)
        for metrics_path in metrics_paths:
            split = split_from_metrics_path(metrics_path)
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            row = {
                **run_meta,
                "split": split,
                "run_dir": str(run_dir),
                "metrics_path": str(metrics_path),
            }
            row.update(numeric_metrics(payload, split))
            rows.append(row)
    return rows


def apply_duplicate_policy(df: pd.DataFrame, policy: str) -> pd.DataFrame:
    keys = ["checkpoint", "dataset", "task", "shots", "split"]
    duplicate_mask = df.duplicated(keys, keep=False)
    if not duplicate_mask.any():
        return df

    duplicates = df.loc[duplicate_mask, keys + ["run_name"]].sort_values(keys)
    if policy == "error":
        rendered = duplicates.to_string(index=False)
        raise ValueError(
            "Duplicate eval cells found. Remove extra runs from the manifest or use "
            f"--duplicate-policy first/latest/mean.\n{rendered}"
        )
    if policy == "first":
        return df.sort_values("run_name").drop_duplicates(keys, keep="first")
    if policy == "latest":
        return df.sort_values(["run_name", "metrics_path"]).drop_duplicates(keys, keep="last")
    if policy == "mean":
        numeric_cols = [
            col
            for col in df.columns
            if col not in keys
            and col not in {"run_name", "model", "task_name", "timestamp", "run_dir", "metrics_path"}
            and pd.api.types.is_numeric_dtype(df[col])
        ]
        grouped = df.groupby(keys, as_index=False)[numeric_cols].mean()
        meta = df.sort_values("run_name").drop_duplicates(keys, keep="last")
        meta_cols = [
            "checkpoint",
            "dataset",
            "task",
            "shots",
            "split",
            "model",
            "task_name",
            "timestamp",
            "run_name",
            "run_dir",
            "metrics_path",
        ]
        return grouped.merge(meta[meta_cols], on=keys, how="left")
    raise ValueError(f"Unsupported duplicate policy: {policy}")


def ordered(values: pd.Series, preferred: list[str] | None = None) -> list[Any]:
    present = list(dict.fromkeys(values.dropna().tolist()))
    if preferred is None:
        return sorted(present)
    preferred_present = [value for value in preferred if value in present]
    rest = sorted(value for value in present if value not in preferred_present)
    return preferred_present + rest


def format_value(value: float) -> str:
    if np.isnan(value):
        return ""
    return f"{value:.3f}"


def plot_heatmap(
    df: pd.DataFrame,
    *,
    metric: str,
    split: str,
    task: str,
    out_path: Path,
    dataset_order: list[str],
) -> bool:
    subset = df[(df["split"] == split) & (df["task"] == task)]
    if subset.empty or metric not in subset.columns or subset[metric].dropna().empty:
        return False

    plot_df = subset.copy()
    plot_df["checkpoint_shot"] = (
        plot_df["checkpoint"].astype(str) + " / " + plot_df["shots"].astype(str) + "shot"
    )
    values = plot_df.pivot(index="dataset", columns="checkpoint_shot", values=metric)
    values = values.reindex(index=ordered(values.index.to_series(), dataset_order))
    column_order = (
        plot_df[["checkpoint", "shots", "checkpoint_shot"]]
        .drop_duplicates()
        .sort_values(["checkpoint", "shots"])["checkpoint_shot"]
        .tolist()
    )
    values = values.reindex(columns=column_order)

    fig_width = max(7.0, 1.15 * len(values.columns) + 2.5)
    fig_height = max(4.5, 0.55 * len(values.index) + 1.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    masked = np.ma.masked_invalid(values.to_numpy(dtype=float))
    image = ax.imshow(masked, cmap="viridis", aspect="auto")

    ax.set_xticks(np.arange(len(values.columns)))
    ax.set_xticklabels(values.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(values.index)))
    ax.set_yticklabels(values.index)
    ax.set_xlabel("checkpoint / shots")
    ax.set_ylabel("dataset")
    ax.set_title(f"{split} {metric} - {TASK_LABELS.get(task, task)}")

    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values.iat[row_idx, col_idx]
            if pd.notna(value):
                ax.text(col_idx, row_idx, format_value(float(value)), ha="center", va="center")

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(metric)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-list", help="Text file containing one eval log directory per line.")
    source.add_argument("--run-glob", help="Glob under --log-root for ad hoc discovery.")
    parser.add_argument("--log-root", default=None, help="Required with --run-glob; optional base for relative run-list entries.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--metrics", default="accuracy,f1,roc_auc")
    parser.add_argument("--splits", default="test")
    parser.add_argument("--tasks", default="nm,lp,pl")
    parser.add_argument("--plot", action="store_true", help="Also write heatmap PNGs.")
    parser.add_argument(
        "--missing-metrics-policy",
        choices=("skip", "error"),
        default="skip",
        help="How to handle eval run directories that have no metrics_*.json files.",
    )
    parser.add_argument(
        "--duplicate-policy",
        choices=("error", "first", "latest", "mean"),
        default="error",
        help="How to handle duplicate checkpoint/dataset/task/shot/split cells.",
    )
    args = parser.parse_args()

    log_root = Path(args.log_root) if args.log_root else None
    if args.run_glob:
        if log_root is None:
            parser.error("--log-root is required with --run-glob")
        run_dirs = discover_runs(log_root, args.run_glob)
    else:
        run_dirs = read_run_list(Path(args.run_list), log_root)

    if not run_dirs:
        raise ValueError("No eval run directories selected.")

    load_table_dependencies()

    datasets = parse_csv(args.datasets)
    metrics = parse_csv(args.metrics)
    splits = parse_csv(args.splits)
    tasks = parse_csv(args.tasks)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_rows(
        run_dirs,
        datasets,
        missing_metrics_policy=args.missing_metrics_policy,
    )
    if not rows:
        raise ValueError("No metric rows found after applying missing-metrics policy.")
    df = pd.DataFrame(rows)
    df = apply_duplicate_policy(df, args.duplicate_policy)
    csv_path = out_dir / "eval_results.csv"
    df.sort_values(["checkpoint", "dataset", "task", "shots", "split"]).to_csv(csv_path, index=False)

    plot_paths: list[Path] = []
    if args.plot:
        load_plot_dependencies()
        for split in splits:
            for task in tasks:
                for metric in metrics:
                    out_path = out_dir / f"heatmap_{split}_{metric}_{task}.png"
                    if plot_heatmap(
                        df,
                        metric=metric,
                        split=split,
                        task=task,
                        out_path=out_path,
                        dataset_order=datasets,
                    ):
                        plot_paths.append(out_path)

    print(f"[done] rows={len(df)}")
    print(f"[done] wrote {csv_path}")
    for path in plot_paths:
        print(f"[done] wrote {path}")
    if args.plot and not plot_paths:
        print("[warn] no plots were produced; check --metrics/--splits/--tasks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
