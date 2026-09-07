#!/usr/bin/env python3
"""Build the current repository evaluation ledger in one-row-per-metric form.

This intentionally reads committed analysis data only. Cluster logs, git history,
branches, and W&B are future ingestion sources.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import OrderedDict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
ANALYSIS_ROOT = ROOT / "scripts" / "experiments" / "analysis"
OUT = Path(__file__).resolve().parent / "data" / "evaluation_ledger.tsv"
WANDB_METADATA = ROOT / "wandb_exports" / "graph_clip_run_metadata.csv"

METRIC_COLUMNS = {
    "roc_auc",
    "roc_auc_ovr_macro",
    "accuracy",
    "f1",
    "f1_macro",
    "spearman",
    "rmse",
    "mae",
    "r2",
    "mse",
    "auc",
    "average_precision",
    "hits_at_50",
    "score",
    "test_accuracy",
    "test_f1",
    "test_roc_auc",
    "accuracy_logged",
    "f1_macro_logged",
    "roc_auc_ovr_macro_logged",
}

COMMON_FIELDS = {
    "model", "model_id", "model_key", "logical_id", "train_id", "dataset",
    "test_graph", "target", "task", "shots", "split", "run", "seed",
    "training_seed", "checkpoint_step", "step", "training_updates", "source",
    "sources", "train_graphs", "group", "arm", "variant", "architecture",
    "component", "baseline", "condition", "n_sources", "order", "rung",
    "added", "added_dataset", "metric", "value", "primary", "negative_kind",
    "scorer", "transform", "n_hop", "feature_view", "source_model",
    "train_task", "eval_task",
}


def read_rows(path: Path):
    delimiter = "\t" if path.suffix == ".tsv" else ","
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if not reader.fieldnames:
            return [], []
        return reader.fieldnames, list(reader)


def load_wandb_index():
    if not WANDB_METADATA.exists():
        return {}, {}
    by_name = {}
    by_id = {}
    with WANDB_METADATA.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("run_name"):
                by_name[row["run_name"]] = row
            if row.get("run_id"):
                by_id[row["run_id"]] = row
    return by_name, by_id


def as_number(value: str):
    if value is None or value == "":
        return None
    text = value.strip()
    if text.lower() in {"nan", "none", "null", "na", "n/a"}:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return number


def clean(value):
    return "" if value is None else str(value).strip()


def first(row, names):
    for name in names:
        value = clean(row.get(name))
        if value:
            return value
    return ""


def infer_task(path: Path, row, metric=""):
    task = first(row, ["task", "eval_task"])
    task = {
        "reg": "regression",
        "pl": "classification",
        "slp": "static_link_prediction",
        "static_lp": "static_link_prediction",
        "node_classification": "classification",
        "node_regression": "regression",
    }.get(task, task)
    if task:
        return task
    text = str(path).lower()
    if "pair_lp" in text:
        return "pair_link_prediction"
    if "static_link_prediction" in text or "slp" in text:
        return "static_link_prediction"
    if "reg_probe" in text or "node_regression" in text or "regression" in text:
        return "regression"
    if "classification" in text or "pl" in text:
        return "classification"
    if metric in {"spearman", "rmse", "mae", "r2", "mse"}:
        return "regression"
    if metric in {"auc", "average_precision", "hits_at_50"}:
        return "pair_link_prediction"
    if metric in {"roc_auc", "roc_auc_ovr_macro", "accuracy", "f1", "f1_macro", "score"}:
        return "classification"
    return ""


def infer_status(path: Path):
    text = str(path).lower()
    if "void" in text:
        return "void_pre_20260723"
    if "baseline" in text or "floor" in text or "random_init" in text:
        return "baseline"
    return "reported"


def infer_metric_columns(fields):
    columns = []
    for field in fields:
        name = field.strip().lower()
        if name in METRIC_COLUMNS:
            columns.append((name, name, None))
        elif name.startswith("test_") and name[5:] in {"accuracy", "f1", "roc_auc"}:
            columns.append((name, name[5:], None))
        elif name.endswith("_std_across_batches"):
            columns.append((name, name.removesuffix("_std_across_batches"), "uncertainty"))
        elif name.endswith("_sample_std"):
            columns.append((name, name.removesuffix("_sample_std"), "uncertainty"))
        elif name.endswith("_std") and name.removesuffix("_std") in METRIC_COLUMNS:
            columns.append((name, name.removesuffix("_std"), "uncertainty"))
    return columns


def candidate(path: Path, fields):
    lower = {field.strip().lower() for field in fields}
    if "metric" in lower and "value" in lower:
        return True
    return bool(infer_metric_columns(fields))


def train_setup(row):
    names = [
        ("group", "group"), ("arm", "arm"), ("variant", "variant"),
        ("source", "source"), ("sources", "sources"),
        ("train_graphs", "train_graphs"), ("architecture", "architecture"),
        ("component", "component"), ("baseline", "baseline"),
        ("condition", "condition"), ("order", "order"), ("rung", "rung"),
    ]
    parts = []
    for field, label in names:
        value = clean(row.get(field))
        if value:
            parts.append(f"{label}={value}")
    return ";".join(parts)


def normalize_metric(name):
    return {
        "roc_auc_ovr_macro_logged": "roc_auc_ovr_macro",
        "f1_macro_logged": "f1_macro",
        "accuracy_logged": "accuracy",
        "test_roc_auc": "roc_auc",
        "test_f1": "f1",
        "test_accuracy": "accuracy",
    }.get(name, name)


def extract_run_time(row):
    """Return (ISO date, ISO timestamp, source, precision) without guessing."""
    candidates = [
        ("run_id", first(row, ["run"])),
        ("run_dir", first(row, ["run_dir"])),
        ("timestamp", first(row, ["timestamp", "datetime", "date"])),
        ("source_log", first(row, ["source_log"])),
        ("evidence_path", first(row, ["evidence_path"])),
    ]
    patterns = [
        (re.compile(r"(?<!\d)(\d{1,2})_(\d{1,2})_(20\d{2})(?:_(\d{1,2})_(\d{1,2})_(\d{1,2}))?(?!\d)"), "dmy_underscore"),
        (re.compile(r"(?<!\d)(20\d{2})-(\d{2})-(\d{2})(?:[T_](\d{2})[:_-](\d{2})[:_-](\d{2}))?(?!\d)"), "ymd"),
    ]
    for source, text in candidates:
        text = clean(text)
        if not text:
            continue
        for pattern, kind in patterns:
            match = pattern.search(text)
            if not match:
                continue
            groups = match.groups()
            if kind == "dmy_underscore":
                day, month, year, hour, minute, second = groups
            else:
                year, month, day, hour, minute, second = groups
            try:
                date_value = f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
                if hour is None:
                    return date_value, "", source, "date"
                timestamp = f"{date_value}T{int(hour):02d}:{int(minute):02d}:{int(second):02d}"
                return date_value, timestamp, source, "timestamp"
            except (TypeError, ValueError):
                continue
    return "", "", "", "unknown"


def match_wandb(row, wandb_by_name, wandb_by_id):
    run_name = first(row, ["run"])
    if run_name and run_name in wandb_by_name:
        return wandb_by_name[run_name], "run_name"
    text = first(row, ["wandb_url", "url"])
    match = re.search(r"wandb\.ai/[^/]+/[^/]+/runs/([A-Za-z0-9]+)", text)
    if match and match.group(1) in wandb_by_id:
        return wandb_by_id[match.group(1)], "url_id"
    return None, ""


def add_record(records, path, row_number, row, metric, value, uncertainty="", wandb_by_name=None, wandb_by_id=None):
    wandb_by_name = wandb_by_name or {}
    wandb_by_id = wandb_by_id or {}
    model = first(row, ["model", "model_id", "model_key", "logical_id", "train_id", "source_model", "model_prefix"])
    dataset = first(row, ["dataset", "test_graph", "test", "graph"])
    target = first(row, ["target"])
    task = infer_task(path, row, normalize_metric(metric))
    if not dataset and task != "regression":
        dataset = first(row, ["target"])
    if not dataset and "covid_task_transfer" in str(path):
        dataset = "covid19_twitter"
    checkpoint = first(row, ["checkpoint_step", "step", "training_updates"])
    shots = first(row, ["shots", "n_shot"])
    seed = first(row, ["seed", "training_seed"])
    run = first(row, ["run"])
    metric = normalize_metric(metric)
    run_date, run_timestamp, date_source, date_precision = extract_run_time(row)
    wandb_row, wandb_match = match_wandb(row, wandb_by_name, wandb_by_id)
    if wandb_row and wandb_row.get("created_at"):
        created_at = wandb_row["created_at"]
        run_date = created_at[:10]
        run_timestamp = created_at
        date_source = "wandb_created_at"
        date_precision = "timestamp"
    status = infer_status(path)
    context = OrderedDict()
    for key in sorted(row):
        if key.lower() in COMMON_FIELDS or key.lower() == metric:
            continue
        if key.lower() == "value" or key.lower().endswith("_std"):
            continue
        val = clean(row.get(key))
        if val:
            context[key] = val
    record = {
        "result_status": status,
        "model_id": model,
        "train_setup": train_setup(row),
        "checkpoint_step": checkpoint,
        "test_dataset": dataset,
        "task": task,
        "target": target,
        "shots": shots,
        "split": first(row, ["split"]),
        "seed": seed,
        "metric": metric,
        "value": f"{value:.17g}",
        "uncertainty": uncertainty,
        "run_id": run,
        "run_date": run_date,
        "run_timestamp": run_timestamp,
        "date_source": date_source,
        "date_precision": date_precision,
        "wandb_run_id": wandb_row.get("run_id", "") if wandb_row else "",
        "wandb_created_at": wandb_row.get("created_at", "") if wandb_row else "",
        "wandb_state": wandb_row.get("state", "") if wandb_row else "",
        "wandb_match": wandb_match,
        "eval_protocol": ";".join(
            f"{key}={clean(row.get(key))}"
            for key in ("negative_kind", "scorer", "transform", "feature_view", "n_hop")
            if clean(row.get(key))
        ),
        "source_path": str(path.relative_to(ROOT)),
        "source_row": str(row_number),
        "context_json": json.dumps(context, separators=(",", ":"), ensure_ascii=False),
    }
    missing = []
    if not model:
        missing.append("model_id")
    if not dataset:
        missing.append("test_dataset")
    if not task:
        missing.append("task")
    record["provenance_quality"] = "complete" if not missing else "missing:" + ",".join(missing)
    key_fields = [record[key] for key in record if key not in {"source_path", "source_row", "context_json"}]
    record["_key"] = hashlib.sha256("\x1f".join(key_fields).encode()).hexdigest()[:16]
    records.setdefault(record["_key"], record)
    existing = records[record["_key"]]
    sources = existing["source_path"].split(" || ")
    if record["source_path"] not in sources:
        existing["source_path"] += " || " + record["source_path"]
        existing["source_row"] += " || " + record["source_row"]


def main():
    records = {}
    source_files = 0
    wandb_by_name, wandb_by_id = load_wandb_index()
    for path in sorted(ANALYSIS_ROOT.rglob("*")):
        if path.suffix not in {".csv", ".tsv"} or "/data/" not in str(path):
            continue
        if path == OUT:
            continue
        fields, rows = read_rows(path)
        if not candidate(path, fields):
            continue
        source_files += 1
        lower_fields = {field.lower() for field in fields}
        for row_number, row in enumerate(rows, start=2):
            if "metric" in lower_fields and "value" in lower_fields:
                metric = clean(row.get("metric"))
                value = as_number(row.get("value"))
                if metric and value is not None:
                    add_record(records, path, row_number, row, metric, value, wandb_by_name=wandb_by_name, wandb_by_id=wandb_by_id)
                continue
            metric_columns = infer_metric_columns(fields)
            uncertainties = {}
            for field, metric, role in metric_columns:
                number = as_number(row.get(field))
                if number is None:
                    continue
                if role == "uncertainty":
                    uncertainties[metric] = f"{number:.17g}"
            for field, metric, role in metric_columns:
                if role == "uncertainty":
                    continue
                value = as_number(row.get(field))
                if value is not None:
                    add_record(records, path, row_number, row, metric, value, uncertainties.get(metric, ""), wandb_by_name, wandb_by_id)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "record_id", "result_status", "provenance_quality", "model_id", "train_setup", "checkpoint_step",
        "test_dataset", "task", "target", "shots", "split", "seed", "metric",
        "value", "uncertainty", "run_id", "run_date", "run_timestamp", "date_source",
        "date_precision", "wandb_run_id", "wandb_created_at", "wandb_state", "wandb_match",
        "eval_protocol", "source_path",
        "source_row", "context_json",
    ]
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for record in sorted(records.values(), key=lambda item: (
            item["model_id"], item["test_dataset"], item["task"], item["target"],
            item["checkpoint_step"], item["metric"], item["value"], item["_key"],
        )):
            record["record_id"] = record.pop("_key")
            writer.writerow(record)
    print(f"source_files={source_files} unique_metric_rows={len(records)} output={OUT}")


if __name__ == "__main__":
    main()
