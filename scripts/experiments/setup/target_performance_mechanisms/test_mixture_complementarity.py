from pathlib import Path
import unittest

import pandas as pd

from .prepare_mixture_complementarity import validate_lattice


class MixtureInventoryTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[4] / "scripts/experiments/analysis/graphs/transfer_prediction/target_performance_mechanisms/data/mixture_complementarity_inputs"
        self.cells = pd.read_csv(root / "classification_long.tsv", sep="\t")
        self.models = pd.read_csv(root / "model_list.tsv", sep="\t")

    def test_complete_frozen_source_combinations(self):
        models = validate_lattice(self.cells, self.models)
        self.assertEqual(models.source_set.map(len).value_counts().to_dict(), {2: 36, 1: 9, 8: 9})
        with self.assertRaises(ValueError):
            validate_lattice(self.cells.iloc[1:], self.models)
        with self.assertRaises(ValueError):
            validate_lattice(self.cells, self.models.iloc[1:])

    def test_protocol_source_and_query_changes_rejected(self):
        for key, value in (("training_seed", 1), ("checkpoint_step", 3000),
                           ("episode_fingerprint", "other episodes"), ("sources", "['covid']"),
                           ("queries", 1), ("roc_auc", float("nan"))):
            bad = self.cells.copy()
            bad.loc[0, key] = value
            with self.assertRaises(ValueError):
                validate_lattice(bad, self.models)

    def test_duplicate_checkpoint_and_composition_rejected(self):
        for key in ("checkpoint", "sources"):
            bad = self.models.copy()
            bad.loc[1, key] = bad.loc[0, key]
            with self.assertRaises(ValueError):
                validate_lattice(self.cells, bad)


if __name__ == "__main__":
    unittest.main()
