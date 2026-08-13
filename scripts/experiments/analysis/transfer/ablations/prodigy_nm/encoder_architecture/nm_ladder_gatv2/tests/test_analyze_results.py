from __future__ import annotations

import importlib.util
import csv
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "analyze_results.py"
SPEC = importlib.util.spec_from_file_location("analyze_results", MODULE_PATH)
assert SPEC and SPEC.loader
analysis = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analysis
SPEC.loader.exec_module(analysis)


class AnalyzeResultsTest(unittest.TestCase):
    def write_run(self, root: Path, rung: int, dataset: str, auc: float, suffix: str, mtime: int) -> Path:
        run_dir = root / (
            f"eval_nm_ladder_gatv2_r{rung}_{rung}src_to_{dataset}_"
            f"nm_3shot_30way_{suffix}"
        )
        data_dir = run_dir / "data"
        data_dir.mkdir(parents=True)
        metrics = data_dir / "metrics_test_step0.json"
        metrics.write_text(json.dumps({"test_roc_auc": auc}), encoding="utf-8")
        os.utime(run_dir, (mtime, mtime))
        return run_dir

    def test_collects_complete_matrix_and_newest_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for rung, _, _ in analysis.RUNGS:
                for column, dataset in enumerate(analysis.CANON):
                    self.write_run(root, rung, dataset, rung / 10 + column / 1000, "old", 100)
            self.write_run(root, 4, "covid_political", 0.9123, "new", 200)

            cells = analysis.collect_cells(root)
            self.assertEqual(len(cells), 64)
            self.assertEqual(analysis.missing_cells(cells), [])
            self.assertAlmostEqual(cells[(4, "covid_political")].auc, 0.9123)

    def test_missing_cells_reports_absence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cells = analysis.collect_cells(Path(tmp))
            self.assertEqual(len(analysis.missing_cells(cells)), 64)

    def test_summary_uses_entry_aligned_delta(self) -> None:
        values = {
            (rung, dataset): rung / 100
            for rung, _, _ in analysis.RUNGS
            for dataset in analysis.CANON
        }
        summary = analysis.summarize(values)
        primary = [event for event in summary["entry_events"] if event["primary"]]
        self.assertEqual(len(primary), 5)
        self.assertTrue(all(abs(event["entry_delta"] - 0.01) < 1e-12 for event in primary))

    def test_write_outputs_keeps_experiment_data_separate(self) -> None:
        values = {
            (rung, dataset): 0.5 + rung / 100
            for rung, _, _ in analysis.RUNGS
            for dataset in analysis.CANON
        }
        sage = {
            (rung, dataset): 0.4 + rung / 100
            for rung, _, _ in analysis.RUNGS
            for dataset in analysis.CANON
        }
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            analysis.write_outputs(values, sage, out_dir)
            with (out_dir / "nm_ladder_backbone_comparison.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 64)
            self.assertEqual(rows[0]["gatv2_minus_sage"], "+0.100000")
            self.assertTrue((out_dir / "nm_ladder_gatv2.csv").is_file())
            self.assertTrue((out_dir / "summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
