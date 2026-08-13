from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("analyze_predictors", HERE / "analyze_predictors.py")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class PredictorAnalysisTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        divergence = __import__("json").loads(MODULE.DEFAULT_DIVERGENCE.read_text())
        cls.graphs = divergence["graphs"]
        cls.divergence = divergence
        transfer = pd.read_csv(MODULE.DEFAULT_TRANSFER)
        cls.auc = MODULE.transfer_matrix(transfer, cls.graphs, "roc_auc")

    def test_complete_nine_by_nine_matrix(self):
        self.assertEqual(self.auc.shape, (9, 9))
        self.assertTrue(np.isfinite(self.auc).all())

    def test_proxy_a_is_strongest_committed_pairwise_predictor(self):
        stats = {}
        for name, matrix in self.divergence["pairwise"].items():
            stats[name] = MODULE.within_target_stat(np.asarray(matrix, float), self.auc)[0]
        self.assertEqual(max(stats, key=lambda x: abs(stats[x])), "proxy_a_distance")
        self.assertAlmostEqual(stats["proxy_a_distance"], -0.7551917106363747)

    def test_homophily_gap_has_no_ranking_signal(self):
        values = np.asarray([self.divergence["per_graph"][g]["feature_homophily"] for g in self.graphs])
        stat = MODULE.within_target_stat(MODULE.scalar_matrix(values, "absolute_gap"), self.auc)[0]
        self.assertAlmostEqual(stat, 0.0026455026455026506)


if __name__ == "__main__":
    unittest.main()
