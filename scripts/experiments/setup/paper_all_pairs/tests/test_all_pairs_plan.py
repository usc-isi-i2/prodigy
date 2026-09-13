import tempfile
import unittest
from pathlib import Path

import yaml

from scripts.experiments.setup.paper_all_pairs.plan import SOURCES, build_pairs, write_configs


class AllPairsPlanTest(unittest.TestCase):
    def test_complete_pair_registry(self):
        pairs = build_pairs()
        self.assertEqual(len(pairs), 36)
        self.assertEqual(len({pair.sources for pair in pairs}), 36)
        for source in SOURCES:
            self.assertEqual(sum(source in pair.sources for pair in pairs), 8)

    def test_generated_configs_preserve_registered_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = write_configs(Path(directory))
            self.assertEqual(len(paths), 36)
            self.assertTrue((Path(directory) / "pairs.tsv").is_file())
            for path in paths:
                config = yaml.safe_load(path.read_text(encoding="utf-8"))
                self.assertEqual(config["dataset_len_cap"] * config["epochs"], 2500)
                self.assertEqual(len(config["neighbor_sampling_source_subset"].split(",")), 2)
                self.assertEqual(config["neighbor_sampling_episode_source_weighting"], "balanced")
                self.assertEqual(config["neighbor_sampling_cross_source_prob"], 0.0)
                self.assertEqual(config["graph_filename"],
                                 "ukr_rus_covid_midterm_all9_facebook_final_core_split_seed0.pt")


if __name__ == "__main__":
    unittest.main()
