#!/usr/bin/env python3
"""Build the canonical long and graph-wide final-experiment result tables."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "AGENTS.md").is_file())
DATA = HERE / "data"
PRODIGY_DATA = DATA / "prodigy_final_core"
sys.path.insert(0, str(REPO / "scripts/experiments/setup/final_core"))

from core_plan import ORDERS, SOURCES  # noqa: E402
from fixed_test_plan import model_for_ladder  # noqa: E402


SAMGPT_COMMIT = "b8bc1223ec65af5d9c71578647c56654b6f81016"
PRODIGY_TRAIN_COMMIT = "fa0db824ad46757841caf38974c5d71c4a5c9757"
PRODIGY_TRAIN_CONFIG = "scripts/experiments/setup/final_core/training.yaml"
PRODIGY_TRAIN_CONFIG_SHA256 = "357f0dfac45b456e665dc394d77b7483933ac131f62cbcfdbd8a1cd0f69263d1"
PRODIGY_TRAIN_PLAN = (
    "scripts/experiments/analysis/transfer/matrices/native_objective/final_core/data/"
    "prodigy_final_core/training/plan.tsv"
)
PRODIGY_FIXED_EVAL_CONFIG = "scripts/experiments/setup/final_core/run_fixed_test_tucker.sh"
PRODIGY_LOGGED_METRICS = (
    "scripts/experiments/analysis/transfer/matrices/native_objective/final_core/data/prodigy_final_core/"
    "log_recovered_metrics/physical_metrics.tsv"
)
PRODIGY_FIXED_EVAL_HASHES = {
    "045ba527ec42b6ca6750d3f1ac1775698496b1b5":
        "3e8346f15121db0fe52283b0efde560ec675f887640990561dfe99ab863b793a",
    "c5be3b9022d0f8638525e138050c11472fe05d60":
        "2a2dffd785c19a6880cbf0891a4e9bbe0f30db60e2bfe4db021bfcf1caf2c31c",
}
SAMGPT_MATRIX_CONFIG = "configs/single_source_nm_matrix/matrix_carc.json"
SAMGPT_MATRIX_CONFIG_SHA256 = "98461691bd50cebbfcebc07244916e0bf0585a1b81856e8ba0be2181b3476df2"
SAMGPT_LADDER_CONFIG = "configs/nm_ladder_9x3/ladder_carc.json"
SAMGPT_LADDER_CONFIG_SHA256 = "ed0ab3eee3c594cffb014e51d78110e11228a37229057350927af6f1831b12dd"
SAMGPT_MATRIX_RESULTS = (
    "scripts/experiments/analysis/transfer/ladders/prodigy_nm/order_and_graph_set/nm_ladder_order_robustness/data/"
    "samgpt_graphcl_specialist_matrix_tucker_h100/metrics_long.csv"
)
SAMGPT_LADDER_RESULTS = (
    "scripts/experiments/analysis/transfer/ladders/prodigy_nm/order_and_graph_set/nm_ladder_order_robustness/data/"
    "samgpt_graphcl_9x3_carc_v100/metrics_long.csv"
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
        "train_run_id": checkpoint_run_id(payload["checkpoint"]),
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


def samgpt_config(component: str) -> tuple[str, str]:
    if component == "matrix":
        return SAMGPT_MATRIX_CONFIG, SAMGPT_MATRIX_CONFIG_SHA256
    return SAMGPT_LADDER_CONFIG, SAMGPT_LADDER_CONFIG_SHA256


def samgpt_base_row(
    *,
    seed_slot: int,
    component: str,
    train_graphs: tuple[str, ...],
    target: str,
    order: str = "",
    rung: int | None = None,
    added_graph: str = "",
) -> dict[str, str]:
    config_path, config_sha = samgpt_config(component)
    design = (
        f"source={train_graphs[0]}" if component == "matrix"
        else f"order={order}|rung={rung}"
    )
    row = blank_row()
    row.update({
        "cell_id": f"samgpt|seed_slot={seed_slot}|{component}|{design}|target={target}",
        "result_status": "observed" if seed_slot == 0 else "pending",
        "architecture": "SAMGPT",
        "component": component,
        "training_seed_slot": str(seed_slot),
        "training_seed": "39" if seed_slot == 0 else "",
        "seed_identity_status": "exact_from_config" if seed_slot == 0 else "pending",
        "order": order,
        "rung": "" if rung is None else str(rung),
        "added_graph": added_graph,
        "train_graphs": graph_list(train_graphs),
        "train_graph_count": str(len(train_graphs)),
        "test_graph": target,
        "test_in_train": format_value(target in train_graphs),
        "train_repo": "samgpt-social",
        "train_commit": SAMGPT_COMMIT,
        "train_config_path": config_path,
        "train_config_sha256": config_sha,
        "train_plan_path": config_path,
        "train_run_id": (
            f"specialist:{train_graphs[0]}" if component == "matrix"
            else f"order:{order}:rung:{rung}"
        ),
        "checkpoint_ref": "",
        "checkpoint_sha256": "",
        "checkpoint_step": "",
        "eval_repo": "samgpt-social",
        "eval_commit": SAMGPT_COMMIT,
        "eval_config_path": config_path,
        "eval_config_sha256": config_sha,
        "eval_protocol": "native_graphcl_discrimination_fixed_unseen_view",
        "eval_view_id": "",
        "eval_seed": "",
        "eval_units": "",
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
    source_path: str,
    source_key: str,
    physical_id: str,
) -> None:
    row.update({
        "eval_view_id": f"eval_seed={source['eval_seed']}|prompt_slot={source['prompt_slot']}",
        "eval_seed": source["eval_seed"],
        "eval_units": source["nodes"],
        "source_result_path": source_path,
        "source_result_key": source_key,
        "physical_result_id": physical_id,
        "primary_value": source["loss"],
        "graphcl_loss": source["loss"],
        "graphcl_accuracy": source["accuracy"],
        "graphcl_positive_probability": source["positive_probability"],
        "graphcl_negative_probability": source["negative_probability"],
        "graphcl_probability_margin": source["probability_margin"],
    })
    if source.get("checkpoint_sha256"):
        row["checkpoint_sha256"] = source["checkpoint_sha256"]


def build_samgpt_rows() -> list[dict[str, str]]:
    matrix_source = read_csv(REPO / SAMGPT_MATRIX_RESULTS)
    ladder_source = read_csv(REPO / SAMGPT_LADDER_RESULTS)
    matrix = {
        (normalize_graph(item["train_source"]), normalize_graph(item["target"])): item
        for item in matrix_source
    }
    ladder = {
        (item["order"], int(item["rung"]), normalize_graph(item["target"])): item
        for item in ladder_source
    }
    if len(matrix) != 81 or len(ladder) != 243:
        raise ValueError("SAMGPT source exports do not match the 81/243 cell contracts")

    rows = []
    for seed_slot in (0, 1, 2):
        for source in GRAPHS:
            for target in GRAPHS:
                row = samgpt_base_row(
                    seed_slot=seed_slot,
                    component="matrix",
                    train_graphs=(source,),
                    target=target,
                )
                if seed_slot == 0:
                    original = matrix[(source, target)]
                    apply_samgpt_metrics(
                        row,
                        original,
                        source_path=SAMGPT_MATRIX_RESULTS,
                        source_key=f"train_source={original['train_source']};target={original['target']}",
                        physical_id=(
                            f"samgpt|seed=39|specialist={source}|target={target}"
                        ),
                    )
                rows.append(row)
        for order, sequence in ORDERS.items():
            for rung in range(1, 10):
                train_graphs = tuple(sequence[:rung])
                for target in GRAPHS:
                    row = samgpt_base_row(
                        seed_slot=seed_slot,
                        component="ladder",
                        train_graphs=train_graphs,
                        target=target,
                        order=order,
                        rung=rung,
                        added_graph=sequence[rung - 1],
                    )
                    if seed_slot == 0:
                        original = ladder[(order, rung, target)]
                        apply_samgpt_metrics(
                            row,
                            original,
                            source_path=SAMGPT_LADDER_RESULTS,
                            source_key=(
                                f"order={order};rung={rung};target={original['target']}"
                            ),
                            physical_id=(
                                f"samgpt|seed=39|order={order}|rung={rung}|target={target}"
                            ),
                        )
                    rows.append(row)
    return rows


def validate_rows(rows: list[dict[str, str]]) -> None:
    ids = [row["cell_id"] for row in rows]
    if len(rows) != 1944 or len(set(ids)) != 1944:
        raise ValueError(f"expected 1,944 unique logical cells, found {len(rows)}/{len(set(ids))}")
    observed = [row for row in rows if row["result_status"] == "observed"]
    pending = [row for row in rows if row["result_status"] == "pending"]
    if len(observed) != 1296 or len(pending) != 648:
        raise ValueError(f"unexpected coverage: observed={len(observed)} pending={len(pending)}")
    if any(row["architecture"] != "SAMGPT" for row in pending):
        raise ValueError("only SAMGPT cells may be pending")
    if any(row["primary_value"] == "" for row in observed):
        raise ValueError("an observed row is missing its primary metric")
    if any(row["primary_value"] != "" for row in pending):
        raise ValueError("a pending row contains an observed primary metric")
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
        "FINAL_RESULT_TABLES_BUILT rows=1944 observed=1296 pending=648 "
        f"sha256={digest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
