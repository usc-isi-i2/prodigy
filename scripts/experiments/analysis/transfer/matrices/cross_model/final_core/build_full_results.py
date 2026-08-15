#!/usr/bin/env python3
"""Build the canonical long and graph-wide final-experiment result tables."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "AGENTS.md").is_file())
DATA = HERE / "data"
PRODIGY_DATA = DATA / "prodigy_final_core"
SAMGPT_DATA = DATA / "samgpt/three_seed"
sys.path.insert(0, str(REPO / "scripts/experiments/setup/final_core"))

from core_plan import ORDERS, SOURCES  # noqa: E402
from fixed_test_plan import model_for_ladder  # noqa: E402


PRODIGY_TRAIN_COMMIT = "fa0db824ad46757841caf38974c5d71c4a5c9757"
PRODIGY_TRAIN_CONFIG = "scripts/experiments/setup/final_core/training.yaml"
PRODIGY_TRAIN_CONFIG_SHA256 = "357f0dfac45b456e665dc394d77b7483933ac131f62cbcfdbd8a1cd0f69263d1"
PRODIGY_TRAIN_PLAN = (
    "scripts/experiments/analysis/transfer/matrices/cross_model/final_core/data/"
    "prodigy_final_core/training/plan.tsv"
)
PRODIGY_FIXED_EVAL_CONFIG = "scripts/experiments/setup/final_core/run_fixed_test_tucker.sh"
PRODIGY_LOGGED_METRICS = (
    "scripts/experiments/analysis/transfer/matrices/cross_model/final_core/data/prodigy_final_core/"
    "log_recovered_metrics/physical_metrics.tsv"
)
PRODIGY_FIXED_EVAL_HASHES = {
    "045ba527ec42b6ca6750d3f1ac1775698496b1b5":
        "3e8346f15121db0fe52283b0efde560ec675f887640990561dfe99ab863b793a",
    "c5be3b9022d0f8638525e138050c11472fe05d60":
        "2a2dffd785c19a6880cbf0891a4e9bbe0f30db60e2bfe4db021bfcf1caf2c31c",
}
SAMGPT_REGISTRY = SAMGPT_DATA / "registry.json"
SAMGPT_CELLS = SAMGPT_DATA / "cells.csv"
SAMGPT_SEEDS = (39, 40, 41)
SAMGPT_TRAIN_CONFIG = "configs/final_core/training.yaml"
SAMGPT_TRAIN_CONFIG_SHA256 = "077e3bc014f7a97845d31d26fac4604d85fe75dcdd82f918e3b26238f8c70bf8"
SAMGPT_EVAL_CONFIG = "scripts/run_final_core_eval_tucker.sh"
SAMGPT_EVAL_CONFIG_SHA256 = "787b4a40635eba8d1f81788a168e264afad5b5d2e247183d87d2101f733de640"
SAMGPT_SOURCE_RESULTS = (
    "scripts/experiments/analysis/transfer/matrices/cross_model/final_core/data/"
    "samgpt/three_seed/cells.csv"
)
SAMGPT_SOURCE_REGISTRY = (
    "scripts/experiments/analysis/transfer/matrices/cross_model/final_core/data/"
    "samgpt/three_seed/registry.json"
)

GRAPH_ALIASES = {
    "ukr_rus_twitter": "ukr_rus",
    "covid19_twitter": "covid",
    "cp_hk_twitter": "cp_hk",
}
GRAPHS = tuple(SOURCES)

LONG_FIELDS = (
    "cell_id",
    "result_status",
    "architecture",
    "component",
    "training_seed_slot",
    "training_seed",
    "seed_identity_status",
    "order",
    "rung",
    "added_graph",
    "train_graphs",
    "train_graph_count",
    "test_graph",
    "test_in_train",
    "train_repo",
    "train_commit",
    "train_config_path",
    "train_config_sha256",
    "train_plan_path",
    "train_run_id",
    "train_run_completed_utc",
    "train_run_time_precision",
    "checkpoint_ref",
    "checkpoint_sha256",
    "checkpoint_step",
    "eval_repo",
    "eval_commit",
    "eval_config_path",
    "eval_config_sha256",
    "eval_protocol",
    "eval_view_id",
    "eval_seed",
    "eval_units",
    "eval_run_completed_utc",
    "eval_run_time_precision",
    "source_result_path",
    "source_result_key",
    "aux_result_path",
    "aux_result_key",
    "physical_result_id",
    "primary_metric",
    "primary_value",
    "nm_accuracy",
    "nm_f1_macro",
    "nm_roc_auc_ovr_macro",
    "nm_f1_auc_source_precision",
    "nm_loss",
    "nm_score_std",
    "nm_auc_replay_accuracy",
    "nm_auc_replay_delta",
    "graphcl_loss",
    "graphcl_accuracy",
    "graphcl_positive_probability",
    "graphcl_negative_probability",
    "graphcl_probability_margin",
    "primary_direction",
)
GRAPH_FIELDS = tuple(
    [f"train:{graph}" for graph in GRAPHS]
    + [f"test:{graph}" for graph in GRAPHS]
)


def repo_relative(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def blank_row() -> dict[str, str]:
    return {field: "" for field in LONG_FIELDS}


def graph_list(graphs: tuple[str, ...] | list[str]) -> str:
    return json.dumps(list(graphs), separators=(",", ":"))


def normalize_graph(name: str) -> str:
    return GRAPH_ALIASES.get(name, name)


def checkpoint_run_id(checkpoint: str) -> str:
    parts = Path(checkpoint).parts
    try:
        return parts[parts.index("final_core") + 1]
    except (ValueError, IndexError):
        return ""


def dated_run_id_utc(run_id: str) -> str:
    match = re.search(r"_(20\d{6})(?:_|$)", run_id)
    if not match:
        raise ValueError(f"run ID does not contain a YYYYMMDD date: {run_id}")
    value = match.group(1)
    return f"{value[:4]}-{value[4:6]}-{value[6:]}"


def prodigy_result_path(seed: int, model_id: str, target: str) -> Path:
    return PRODIGY_DATA / "fixed_test/results" / f"seed_{seed}" / model_id / f"{target}.json"


def prodigy_auc_path(seed: int, source: str, target: str) -> Path:
    return PRODIGY_DATA / "auc/results" / f"seed_{seed}" / f"ss_{source}" / f"{target}.json"


def prodigy_row(
    *,
    seed: int,
    component: str,
    train_graphs: tuple[str, ...],
    target: str,
    model_id: str,
    order: str = "",
    rung: int | None = None,
    added_graph: str = "",
    logged_metric: dict[str, str],
) -> dict[str, str]:
    result_path = prodigy_result_path(seed, model_id, target)
    payload = load_json(result_path)
    eval_commit = payload["evaluation_commit"]
    if eval_commit not in PRODIGY_FIXED_EVAL_HASHES:
        raise ValueError(f"{result_path}: unregistered evaluation commit {eval_commit}")
    design = (
        f"source={train_graphs[0]}" if component == "matrix"
        else f"order={order}|rung={rung}"
    )
    cell_id = f"prodigy|seed_slot={seed}|{component}|{design}|target={target}"
    physical_id = (
        f"prodigy|seed={seed}|model={model_id}|target={target}|"
        f"checkpoint={payload['checkpoint_step']}"
    )
    if logged_metric["physical_result_id"] != physical_id:
        raise ValueError(
            f"{result_path}: logged metric key mismatch "
            f"{logged_metric['physical_result_id']} versus {physical_id}"
        )
    logged_delta = float(logged_metric["accuracy_logged"]) - float(payload["score"])
    if abs(logged_delta) > 5.1e-5:
        raise ValueError(
            f"{result_path}: logged accuracy differs from result by {logged_delta}"
        )
    row = blank_row()
    train_run_id = checkpoint_run_id(payload["checkpoint"])
    row.update({
        "cell_id": cell_id,
        "result_status": "observed",
        "architecture": "PRODIGY",
        "component": component,
        "training_seed_slot": str(seed),
        "training_seed": str(seed),
        "seed_identity_status": "exact",
        "order": order,
        "rung": "" if rung is None else str(rung),
        "added_graph": added_graph,
        "train_graphs": graph_list(train_graphs),
        "train_graph_count": str(len(train_graphs)),
        "test_graph": target,
        "test_in_train": format_value(target in train_graphs),
        "train_repo": "prodigy",
        "train_commit": PRODIGY_TRAIN_COMMIT,
        "train_config_path": PRODIGY_TRAIN_CONFIG,
        "train_config_sha256": PRODIGY_TRAIN_CONFIG_SHA256,
        "train_plan_path": PRODIGY_TRAIN_PLAN,
        "train_run_id": train_run_id,
        "train_run_completed_utc": dated_run_id_utc(train_run_id),
        "train_run_time_precision": "date_from_run_id",
        "checkpoint_ref": payload["checkpoint"],
        "checkpoint_sha256": "",
        "checkpoint_step": format_value(payload["checkpoint_step"]),
        "eval_repo": "prodigy",
        "eval_commit": eval_commit,
        "eval_config_path": PRODIGY_FIXED_EVAL_CONFIG,
        "eval_config_sha256": PRODIGY_FIXED_EVAL_HASHES[eval_commit],
        "eval_protocol": payload["protocol"],
        "eval_view_id": payload["observed_episode_fingerprint"],
        "eval_seed": "",
        "eval_units": format_value(payload["episode_count"]),
        "eval_run_completed_utc": payload["created_utc"],
        "eval_run_time_precision": "timestamp_from_result",
        "source_result_path": repo_relative(result_path),
        "source_result_key": "json_root",
        "aux_result_path": PRODIGY_LOGGED_METRICS,
        "aux_result_key": physical_id,
        "physical_result_id": physical_id,
        "primary_metric": "neighbor_matching_accuracy",
        "primary_value": format_value(payload["score"]),
        "primary_direction": "maximize",
        "nm_accuracy": format_value(payload["score"]),
        "nm_f1_macro": logged_metric["f1_macro_logged"],
        "nm_roc_auc_ovr_macro": logged_metric["roc_auc_ovr_macro_logged"],
        "nm_f1_auc_source_precision": "fixed_test_stdout_4_decimal",
        "nm_loss": format_value(payload["loss"]),
        "nm_score_std": format_value(payload["score_std"]),
    })
    if component == "matrix":
        source = train_graphs[0]
        auc_path = prodigy_auc_path(seed, source, target)
        auc = load_json(auc_path)
        replay_delta = float(auc["accuracy"]) - float(payload["score"])
        row.update({
            "aux_result_path": repo_relative(auc_path),
            "aux_result_key": "json_root",
            "nm_f1_macro": format_value(auc["f1_macro"]),
            "nm_roc_auc_ovr_macro": format_value(auc["roc_auc_ovr_macro"]),
            "nm_f1_auc_source_precision": "auc_replay_json_full_precision",
            "nm_auc_replay_accuracy": format_value(auc["accuracy"]),
            "nm_auc_replay_delta": format_value(replay_delta),
        })
    return row


def build_prodigy_rows() -> list[dict[str, str]]:
    logged_rows = read_tsv(REPO / PRODIGY_LOGGED_METRICS)
    logged = {
        (int(row["seed"]), row["model_id"], row["target"]): row
        for row in logged_rows
    }
    if len(logged_rows) != 837 or len(logged) != 837:
        raise ValueError(
            f"logged PRODIGY metrics do not match 837 physical cells: "
            f"rows={len(logged_rows)} unique={len(logged)}"
        )
    rows = []
    for seed in (0, 1, 2):
        for source in GRAPHS:
            for target in GRAPHS:
                rows.append(prodigy_row(
                    seed=seed,
                    component="matrix",
                    train_graphs=(source,),
                    target=target,
                    model_id=f"ss_{source}",
                    logged_metric=logged[(seed, f"ss_{source}", target)],
                ))
        for order, sequence in ORDERS.items():
            for rung in range(1, 10):
                train_graphs = tuple(sequence[:rung])
                model_id = model_for_ladder(order, rung).model_id
                for target in GRAPHS:
                    rows.append(prodigy_row(
                        seed=seed,
                        component="ladder",
                        train_graphs=train_graphs,
                        target=target,
                        model_id=model_id,
                        order=order,
                        rung=rung,
                        added_graph=sequence[rung - 1],
                        logged_metric=logged[(seed, model_id, target)],
                    ))
    return rows


def samgpt_base_row(
    *,
    seed_slot: int,
    seed: int,
    component: str,
    train_graphs: tuple[str, ...],
    target: str,
    registry: dict[str, Any],
    order: str = "",
    rung: int | None = None,
    added_graph: str = "",
) -> dict[str, str]:
    design = (
        f"source={train_graphs[0]}" if component == "matrix"
        else f"order={order}|rung={rung}"
    )
    row = blank_row()
    row.update({
        "cell_id": f"samgpt|seed_slot={seed_slot}|{component}|{design}|target={target}",
        "result_status": "observed",
        "architecture": "SAMGPT",
        "component": component,
        "training_seed_slot": str(seed_slot),
        "training_seed": str(seed),
        "seed_identity_status": "exact_from_config",
        "order": order,
        "rung": "" if rung is None else str(rung),
        "added_graph": added_graph,
        "train_graphs": graph_list(train_graphs),
        "train_graph_count": str(len(train_graphs)),
        "test_graph": target,
        "test_in_train": format_value(target in train_graphs),
        "train_repo": "samgpt-social",
        "train_commit": registry["training_commit"],
        "train_config_path": SAMGPT_TRAIN_CONFIG,
        "train_config_sha256": SAMGPT_TRAIN_CONFIG_SHA256,
        "train_plan_path": SAMGPT_TRAIN_CONFIG,
        "train_run_id": "",
        "train_run_completed_utc": registry["training_completed_utc"],
        "train_run_time_precision": "timestamp_from_training_manifest_path",
        "checkpoint_ref": "",
        "checkpoint_sha256": "",
        "checkpoint_step": str(registry["terminal_checkpoint_update"]),
        "eval_repo": "samgpt-social",
        "eval_commit": registry["evaluation_commit"],
        "eval_config_path": SAMGPT_EVAL_CONFIG,
        "eval_config_sha256": SAMGPT_EVAL_CONFIG_SHA256,
        "eval_protocol": registry["protocol"],
        "eval_view_id": "",
        "eval_seed": "",
        "eval_units": "",
        "eval_run_completed_utc": registry["evaluation_completed_date_utc"],
        "eval_run_time_precision": "date_from_terminal_result_mtime_audit",
        "source_result_path": "",
        "source_result_key": "",
        "aux_result_path": "",
        "aux_result_key": "",
        "physical_result_id": "",
        "primary_metric": "graphcl_bce_loss",
        "primary_value": "",
        "primary_direction": "minimize",
    })
    return row


def apply_samgpt_metrics(
    row: dict[str, str],
    source: dict[str, str],
    *,
    registry: dict[str, Any],
) -> None:
    seed = int(source["seed"])
    target = normalize_graph(source["target"])
    prompt_slot = GRAPHS.index(target)
    physical_id = (
        f"samgpt|seed={seed}|model={source['model_id']}|target={target}|"
        f"checkpoint={source['checkpoint_update']}"
    )
    row.update({
        "train_run_id": f"seed_{seed}/{source['model_id']}",
        "checkpoint_ref": (
            f"samgpt-social@{registry['evidence_commit']}:seed_{seed}/"
            f"{source['model_id']}/checkpoint_update_{source['checkpoint_update']}"
        ),
        "checkpoint_step": source["checkpoint_update"],
        "eval_view_id": f"eval_seed={30000 + prompt_slot}|prompt_slot={prompt_slot}",
        "eval_seed": str(30000 + prompt_slot),
        "source_result_path": SAMGPT_SOURCE_RESULTS,
        "source_result_key": physical_id,
        "aux_result_path": SAMGPT_SOURCE_REGISTRY,
        "aux_result_key": registry["evidence_commit"],
        "physical_result_id": physical_id,
        "primary_value": source["loss"],
        "graphcl_loss": source["loss"],
        "graphcl_accuracy": source["accuracy"],
        "graphcl_positive_probability": source["positive_probability"],
        "graphcl_negative_probability": source["negative_probability"],
        "graphcl_probability_margin": source["probability_margin"],
    })
def verify_samgpt_import(registry: dict[str, Any]) -> list[dict[str, str]]:
    if tuple(registry["training_seeds"]) != SAMGPT_SEEDS:
        raise ValueError("SAMGPT registry does not contain the registered training seeds")
    if registry["terminal_checkpoint_update"] != 500:
        raise ValueError("SAMGPT registry terminal checkpoint is not update 500")
    if registry["training_completed_utc"] != "2026-08-07T18:53:24Z":
        raise ValueError("SAMGPT registry has an unexpected training completion time")
    if registry["evaluation_completed_date_utc"] != "2026-08-08":
        raise ValueError("SAMGPT registry has an unexpected evaluation completion date")
    for name, record in registry["files"].items():
        path = SAMGPT_DATA / name
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != record["sha256"]:
            raise ValueError(f"{path}: SHA-256 mismatch")
    cells = read_csv(SAMGPT_CELLS)
    if len(cells) != registry["files"]["cells.csv"]["rows"]:
        raise ValueError("SAMGPT cells.csv row count disagrees with its registry")
    return cells


def samgpt_source_key(source: dict[str, str]) -> tuple[int, frozenset[str], str]:
    graphs = frozenset(normalize_graph(item) for item in source["sources"].split(","))
    return int(source["seed"]), graphs, normalize_graph(source["target"])


def build_samgpt_rows() -> list[dict[str, str]]:
    registry = load_json(SAMGPT_REGISTRY)
    cells = verify_samgpt_import(registry)
    terminal = [
        row
        for row in cells
        if int(row["checkpoint_update"]) == registry["terminal_checkpoint_update"]
    ]
    physical = {samgpt_source_key(row): row for row in terminal}
    if len(terminal) != 837 or len(physical) != 837:
        raise ValueError(
            f"SAMGPT terminal source does not match 3 x 31 x 9 physical cells: "
            f"rows={len(terminal)} unique={len(physical)}"
        )

    rows = []
    for seed_slot, seed in enumerate(SAMGPT_SEEDS):
        for source in GRAPHS:
            for target in GRAPHS:
                row = samgpt_base_row(
                    seed_slot=seed_slot,
                    seed=seed,
                    component="matrix",
                    train_graphs=(source,),
                    target=target,
                    registry=registry,
                )
                original = physical[(seed, frozenset((source,)), target)]
                apply_samgpt_metrics(row, original, registry=registry)
                rows.append(row)
        for order, sequence in ORDERS.items():
            for rung in range(1, 10):
                train_graphs = tuple(sequence[:rung])
                for target in GRAPHS:
                    row = samgpt_base_row(
                        seed_slot=seed_slot,
                        seed=seed,
                        component="ladder",
                        train_graphs=train_graphs,
                        target=target,
                        registry=registry,
                        order=order,
                        rung=rung,
                        added_graph=sequence[rung - 1],
                    )
                    original = physical[(seed, frozenset(train_graphs), target)]
                    apply_samgpt_metrics(row, original, registry=registry)
                    rows.append(row)
    return rows


def validate_rows(rows: list[dict[str, str]]) -> None:
    ids = [row["cell_id"] for row in rows]
    if len(rows) != 1944 or len(set(ids)) != 1944:
        raise ValueError(f"expected 1,944 unique logical cells, found {len(rows)}/{len(set(ids))}")
    observed = [row for row in rows if row["result_status"] == "observed"]
    pending = [row for row in rows if row["result_status"] == "pending"]
    if len(observed) != 1944 or pending:
        raise ValueError(f"unexpected coverage: observed={len(observed)} pending={len(pending)}")
    if any(row["primary_value"] == "" for row in observed):
        raise ValueError("an observed row is missing its primary metric")
    if any(not row["train_run_completed_utc"] or not row["eval_run_completed_utc"] for row in observed):
        raise ValueError("an observed row is missing its training or evaluation run date")
    prodigy = [row for row in observed if row["architecture"] == "PRODIGY"]
    if any(not row["nm_roc_auc_ovr_macro"] for row in prodigy):
        raise ValueError("an observed PRODIGY row is missing ROC-AUC")


def write_tsv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def wide_row(row: dict[str, str]) -> dict[str, str]:
    result = dict(row)
    train_graphs = set(json.loads(row["train_graphs"]))
    for graph in GRAPHS:
        result[f"train:{graph}"] = "1" if graph in train_graphs else "0"
        result[f"test:{graph}"] = "1" if graph == row["test_graph"] else "0"
    return result


def row_sort_key(row: dict[str, str]) -> tuple[Any, ...]:
    graph_rank = {graph: index for index, graph in enumerate(GRAPHS)}
    architecture_rank = {"PRODIGY": 0, "SAMGPT": 1}
    component_rank = {"matrix": 0, "ladder": 1}
    if row["component"] == "matrix":
        design_key = (graph_rank[json.loads(row["train_graphs"])[0]], 0)
    else:
        design_key = (("A", "B", "C").index(row["order"]), int(row["rung"]))
    return (
        architecture_rank[row["architecture"]],
        int(row["training_seed_slot"]),
        component_rank[row["component"]],
        *design_key,
        graph_rank[row["test_graph"]],
    )


def main() -> int:
    rows = build_prodigy_rows() + build_samgpt_rows()
    rows.sort(key=row_sort_key)
    validate_rows(rows)
    write_tsv(DATA / "results_full_long.tsv", LONG_FIELDS, rows)
    write_tsv(
        DATA / "results_full_graphwide.tsv",
        (*LONG_FIELDS, *GRAPH_FIELDS),
        [wide_row(row) for row in rows],
    )
    digest = hashlib.sha256((DATA / "results_full_long.tsv").read_bytes()).hexdigest()
    print(
        "FINAL_RESULT_TABLES_BUILT rows=1944 observed=1944 pending=0 "
        f"sha256={digest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
