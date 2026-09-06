import copy
import json
from pathlib import Path
import unittest

import yaml

from .audit_member_contract import check_episode_sources, check_protocol


class ContractTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).parent
        self.inventory = json.loads((root / "data/member_training_config_inventory.json").read_text())
        self.protocol = json.loads((root / "data/source_sampler_protocol.json").read_text())
        folder = root.parents[5] / "scripts/experiments/setup/target_performance_mechanisms/member_configs"
        self.declared = {p["prefix"]: p for f in folder.glob("*.yaml") for p in [yaml.safe_load(f.read_text())]}

    def test_all_recorded_configurations_match(self):
        actual = check_protocol(self.inventory, self.declared, self.protocol)
        self.assertEqual(len(actual), 24)

    def test_changed_shared_recipe_or_single_arm_fails(self):
        self.inventory["base_effective_config"]["learning_rate"] = .003
        with self.assertRaisesRegex(ValueError, "declared CPU recipe"):
            check_protocol(self.inventory, self.declared, self.protocol)
        self.setUp()
        self.inventory["effective_configs"][1]["changes"]["ignore_label_embeddings"] = False
        with self.assertRaisesRegex(ValueError, "non-treatment"):
            check_protocol(self.inventory, self.declared, self.protocol)

    def test_launch_mismatch_or_wrong_graph_fails(self):
        self.inventory["planned_jobs"][1]["changes"]["learning_rate"] = .004
        with self.assertRaisesRegex(ValueError, "launch manifest"):
            check_protocol(self.inventory, self.declared, self.protocol)
        self.setUp()
        self.protocol["graph_path"] = "/wrong/graph.pt"
        with self.assertRaisesRegex(ValueError, "different graph"):
            check_protocol(self.inventory, self.declared, self.protocol)

    def test_source_labels_must_match_intended_source_and_count(self):
        actual = check_protocol(self.inventory, self.declared, self.protocol)
        arms = []
        for p in actual:
            source = p["neighbor_sampling_source_subset"]
            arms.append({"model_id": p["prefix"], "source": source, "seed": p["seed"],
                         "policy": p["neighbor_matching_member_policy"],
                         "summary": {"episodes": 10000, "source_ids": {str(self.protocol["source_names"].index(source)): 10000}}})
        check_episode_sources(arms, actual, self.protocol)
        for counts in ({"7": 10000}, {"0": 9999}, {"0": 9999, "7": 1}):
            altered = copy.deepcopy(arms)
            altered[0]["summary"]["source_ids"] = counts
            with self.assertRaisesRegex(ValueError, "source labels"):
                check_episode_sources(altered, actual, self.protocol)


if __name__ == "__main__":
    unittest.main()
