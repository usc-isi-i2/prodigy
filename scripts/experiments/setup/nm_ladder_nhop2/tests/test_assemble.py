import importlib.util
import csv
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[5]
ASSEMBLER_PATH = (
    REPO / "scripts" / "experiments" / "analysis" / "nm_ladder_nhop2" / "assemble_results.py"
)
PLOT_PATH = (
    REPO / "scripts" / "experiments" / "analysis" / "nm_ladder_nhop2" / "plot_nhop_comparison.py"
)


def load_assembler():
    spec = importlib.util.spec_from_file_location("nm_ladder_nhop2_assembler", ASSEMBLER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_plotter():
    spec = importlib.util.spec_from_file_location("nm_ladder_nhop2_plot", PLOT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_synthetic_logs_map_21_models_to_24_rows(tmp_path):
    assembler = load_assembler()
    expected = {}
    for model_index, row in enumerate(assembler.PLAN.unique_rows()):
        prefix = str(row["prefix"])
        expected[prefix] = {}
        for dataset_index, dataset in enumerate(assembler.DATASETS):
            value = 0.5 + model_index / 1000 + dataset_index / 10000
            expected[prefix][dataset] = value
            run = tmp_path / f"eval_{prefix}_to_{dataset}_nm_3shot_30way_20260801"
            data = run / "data"
            data.mkdir(parents=True)
            (data / "metrics_test_step0.json").write_text(
                json.dumps({"test_roc_auc": value}), encoding="utf-8"
            )

    wide, long_rows, missing = assembler.assemble(tmp_path)
    assert missing == []
    assert len(wide) == 24
    assert len(long_rows) == 192

    b2 = next(row for row in wide if row["order"] == "B" and row["rung"] == 2)
    a2 = next(row for row in wide if row["order"] == "A" and row["rung"] == 2)
    assert b2["model_prefix"] == a2["model_prefix"]
    assert all(b2[dataset] == a2[dataset] for dataset in assembler.DATASETS)

    hop1 = {
        (str(row["order"]), int(row["rung"]), str(row["test_graph"])): float(row["auc"]) - 0.01
        for row in long_rows
    }
    paired = assembler.paired_rows(long_rows, hop1)
    assert len(paired) == 192
    assert all(abs(float(row["delta_h2_minus_h1"]) - 0.01) < 1e-12 for row in paired)


def test_order_a_can_be_assembled_without_robustness_models(tmp_path):
    assembler = load_assembler()
    prefixes = {
        str(row["prefix"])
        for row in assembler.PLAN.plan()
        if row["order"] == "A"
    }
    for model_index, prefix in enumerate(sorted(prefixes)):
        for dataset_index, dataset in enumerate(assembler.DATASETS):
            run = tmp_path / f"eval_{prefix}_to_{dataset}_nm_3shot_30way_20260801"
            data = run / "data"
            data.mkdir(parents=True)
            value = 0.5 + model_index / 1000 + dataset_index / 10000
            (data / "metrics_test_step0.json").write_text(
                json.dumps({"test_roc_auc": value}), encoding="utf-8"
            )

    wide, long_rows, missing = assembler.assemble(tmp_path, orders={"A"})
    assert missing == []
    assert len(wide) == 8
    assert len(long_rows) == 64
    assert {row["order"] for row in wide} == {"A"}


def test_plotter_renders_complete_paired_table(tmp_path):
    assembler = load_assembler()
    plotter = load_plotter()
    data_path = tmp_path / "comparison.csv"
    fieldnames = [
        "order", "rung", "test_graph", "entry_rung", "rel_to_entry",
        "in_training", "auc_h1", "auc_h2", "delta_h2_minus_h1",
        "model_prefix_h2m",
    ]
    rows = []
    for plan_row in assembler.PLAN.plan():
        order = str(plan_row["order"])
        rung = int(plan_row["rung"])
        for dataset_index, dataset in enumerate(assembler.DATASETS):
            source_key = assembler.KEY_OF_DATASET[dataset]
            entry = assembler.PLAN.ORDERS[order].index(source_key) + 1
            h1 = 0.70 + rung / 100 + dataset_index / 1000
            h2 = h1 + 0.01
            rows.append(
                {
                    "order": order,
                    "rung": rung,
                    "test_graph": dataset,
                    "entry_rung": entry,
                    "rel_to_entry": rung - entry,
                    "in_training": int(rung >= entry),
                    "auc_h1": h1,
                    "auc_h2": h2,
                    "delta_h2_minus_h1": 0.01,
                    "model_prefix_h2m": plan_row["prefix"],
                }
            )
    with data_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    plotter.DATA = data_path
    plotter.FIGURES = tmp_path / "figures"
    plotter.main()
    assert (plotter.FIGURES / "nm_ladder_nhop_comparison.pdf").is_file()
    assert (plotter.FIGURES / "nm_ladder_nhop_comparison.png").is_file()


def test_plotter_renders_order_a_subset(tmp_path):
    assembler = load_assembler()
    plotter = load_plotter()
    data_path = tmp_path / "comparison_A.csv"
    fieldnames = [
        "order", "rung", "test_graph", "entry_rung", "rel_to_entry",
        "in_training", "auc_h1", "auc_h2", "delta_h2_minus_h1",
        "model_prefix_h2m",
    ]
    rows = []
    for plan_row in assembler.PLAN.plan():
        if plan_row["order"] != "A":
            continue
        rung = int(plan_row["rung"])
        for dataset_index, dataset in enumerate(assembler.DATASETS):
            source_key = assembler.KEY_OF_DATASET[dataset]
            entry = assembler.PLAN.ORDERS["A"].index(source_key) + 1
            h1 = 0.70 + rung / 100 + dataset_index / 1000
            rows.append(
                {
                    "order": "A",
                    "rung": rung,
                    "test_graph": dataset,
                    "entry_rung": entry,
                    "rel_to_entry": rung - entry,
                    "in_training": int(rung >= entry),
                    "auc_h1": h1,
                    "auc_h2": h1 + 0.01,
                    "delta_h2_minus_h1": 0.01,
                    "model_prefix_h2m": plan_row["prefix"],
                }
            )
    with data_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    plotter.DATA_A = data_path
    plotter.FIGURES = tmp_path / "figures"
    plotter.main("A")
    assert (plotter.FIGURES / "nm_ladder_nhop_comparison_order_A.pdf").is_file()
    assert (plotter.FIGURES / "nm_ladder_nhop_comparison_order_A.png").is_file()
