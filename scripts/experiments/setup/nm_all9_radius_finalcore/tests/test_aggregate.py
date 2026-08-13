import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import aggregate_radius_results  # noqa: E402
from radius_plan import ARMS, PANELS  # noqa: E402


def test_one_seed_aggregation(monkeypatch, tmp_path):
    results_root = tmp_path / "results"
    summary_root = tmp_path / "summary"
    for arm in ARMS:
        directory = results_root / "seed_0" / arm.arm_id
        directory.mkdir(parents=True)
        selection = {
            "created_utc": "selection-time",
            "selected_checkpoint_step": 10000,
            "checkpoint_steps": [2500, 5000, 7500, 10000],
            "validation_panels": [panel.panel_id for panel in PANELS],
            "validation_results": [
                {
                    "checkpoint_step": step,
                    "panel": panel,
                    "score": 0.5,
                    "score_std": 0.1,
                    "loss": 1.0,
                }
                for step in (2500, 5000, 7500, 10000)
                for panel in ("radius2", "radius3", "global", "within_source")
            ],
        }
        result = {
            "selection_created_utc": "selection-time",
            "selected_checkpoint_step": 10000,
            "test_results": [
                {
                    "panel": panel.panel_id,
                    "score": 0.6,
                    "score_std": 0.1,
                    "loss": 0.9,
                }
                for panel in PANELS
            ],
        }
        (directory / "selection.json").write_text(json.dumps(selection))
        (directory / "result.json").write_text(json.dumps(result))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "aggregate_radius_results.py",
            "--results-root",
            str(results_root),
            "--output-root",
            str(summary_root),
            "--seeds",
            "0",
        ],
    )
    assert aggregate_radius_results.main() == 0
    summary = json.loads((summary_root / "summary.json").read_text())
    assert summary["seeds"] == [0]
    assert all(row["seed_std"] == 0.0 for row in summary["summary"])
