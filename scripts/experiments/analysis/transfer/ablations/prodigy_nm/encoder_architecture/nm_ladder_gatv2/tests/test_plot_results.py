from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "plot_results.py"
SPEC = importlib.util.spec_from_file_location("plot_results", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PLOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLOT)


class PlotResultsTest(unittest.TestCase):
    def test_registered_entry_checks(self) -> None:
        matrix = PLOT.load_gatv2_matrix()
        self.assertEqual(matrix.shape, (8, 8))
        deltas = PLOT.entry_deltas(matrix)
        primary = deltas[2:]
        self.assertTrue(np.all(primary > 0))
        self.assertTrue(np.all(primary[[0, 1, 2, 4]] > 0.02))

    def test_backbone_pairing_is_complete_and_close(self) -> None:
        rows = PLOT.load_comparison()
        self.assertEqual(len(rows), 64)
        sage = np.array([float(row["sage_auc"]) for row in rows])
        gatv2 = np.array([float(row["gatv2_auc"]) for row in rows])
        self.assertGreater(float(np.corrcoef(sage, gatv2)[0, 1]), 0.99)
        self.assertAlmostEqual(float(np.mean(gatv2 - sage)), -0.009651, places=5)


if __name__ == "__main__":
    unittest.main()
