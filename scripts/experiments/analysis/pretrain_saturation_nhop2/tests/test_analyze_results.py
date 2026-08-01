from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve()
MODULE_PATH = HERE.parents[1] / "analyze_results.py"
SPEC = importlib.util.spec_from_file_location("sat_h2_analysis", MODULE_PATH)
analysis = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(analysis)


def write_classification_matrix(root: Path) -> None:
    for model in analysis.expected_raw_models():
        for dataset in analysis.CLASSIFICATION_DATASETS:
            run = root / f"eval_{model}_to_{dataset}_pl_10shot_20260801"
            data = run / "data"
            data.mkdir(parents=True)
            (data / "metrics_test_step0.json").write_text(
                json.dumps({"test_roc_auc": 0.75}), encoding="utf-8"
            )


def write_probe_matrix(root: Path) -> None:
    fields = [
        "dataset", "model", "target", "alpha", "features", "spearman",
        "rmse", "r2", "n_pred", "n_labeled",
    ]
    for dataset in analysis.REGRESSION_DATASETS:
        path = root / f"{dataset}__reg_probe.csv"
        rows = []
        for target in analysis.TARGETS:
            rows.append({
                "dataset": dataset, "model": "__features_only__", "target": target,
                "alpha": 1.0, "features": "raw_x", "spearman": 0.1,
                "rmse": 1, "r2": 0, "n_pred": 10, "n_labeled": 100,
            })
            for model in analysis.expected_raw_models():
                rows.append({
                    "dataset": dataset, "model": model, "target": target,
                    "alpha": 1.0, "features": "frozen_emb", "spearman": 0.2,
                    "rmse": 1, "r2": 0, "n_pred": 10, "n_labeled": 100,
                })
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def test_collectors_require_and_return_complete_raw_matrices(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    probe = tmp_path / "probe"
    logs.mkdir(); probe.mkdir()
    write_classification_matrix(logs)
    write_probe_matrix(probe)

    classification = analysis.collect_classification(logs)
    regression, floors = analysis.collect_regression(probe)
    assert len(classification) == 19 * 4
    assert len(regression) == 19 * 4 * 2
    assert len(floors) == 4 * 2


def test_shared_step0_expands_to_three_arms() -> None:
    import pandas as pd

    raw = pd.DataFrame([{
        "model": analysis.SHARED_STEP0,
        "dataset": "covid_political",
        "target": "",
        "value": 0.5,
        "evidence_path": "metric.json",
    }])
    expanded = analysis.expand_shared_step0(raw, "classification", "roc_auc")
    assert set(expanded.arm) == set(analysis.ARMS)
    assert set(expanded.step) == {0}
    assert expanded.shared_step0.all()


def test_main_builds_paired_outputs(tmp_path: Path, monkeypatch) -> None:
    import pandas as pd

    logs = tmp_path / "logs"
    probe = tmp_path / "probe"
    out = tmp_path / "out"
    logs.mkdir(); probe.mkdir()
    write_classification_matrix(logs)
    write_probe_matrix(probe)
    monkeypatch.setattr(analysis, "make_figure", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(sys, "argv", [
        "analyze_results.py",
        "--log-root", str(logs),
        "--probe-dir", str(probe),
        "--out-dir", str(out),
    ])
    assert analysis.main() == 0
    long = pd.read_csv(out / "pretrain_saturation_nhop2_long.csv")
    comparison = pd.read_csv(out / "nhop_comparison.csv")
    assert len(long) == 3 * 7 * (4 + 4 * 2)
    assert len(comparison) == len(long)
    assert comparison.value_h1.notna().all()
