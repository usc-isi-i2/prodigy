import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest

import torch

from experiments.episode_audit import digest
from .make_member_configs import POLICIES
from .verify_member_training import audit_consumed, check_groups, model_digest


class MemberVerifierTest(unittest.TestCase):
    def test_audit_validates_counts_and_hashes(self):
        ids = torch.tensor([[list(range(7)), list(range(5, 12))]])
        anchor = torch.tensor([[42, 43]])
        row = {"step": 1, "member_ids": ids.tolist(), "anchor_ids": anchor.tolist(),
               "query_roles": [0, 0, 0, 1, 1, 1, 1], "source_ids": [0],
               "context_node_counts": torch.full_like(ids, 3).tolist(),
               "anchor_sha256": digest(anchor), "member_set_sha256": digest(ids.sort(-1).values),
               "member_order_sha256": digest(ids)}
        params = {"n_way": 2, "n_shots": 3, "n_query": 4, "batch_size": 1}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.jsonl.gz"
            with gzip.open(path, "wt") as handle:
                handle.write(json.dumps(row) + "\n")
            result = audit_consumed(path, params, 1)["summary"]
            self.assertEqual(result["unique_members"], 12)
            self.assertAlmostEqual(result["effective_member_count"], 196 / 18)
            self.assertAlmostEqual(result["cross_class_duplicate_fraction"], 2 / 14)
            self.assertEqual(result["context_nodes_mean"], 3)
            with self.assertRaisesRegex(ValueError, "incomplete"):
                audit_consumed(path, params, 2)
            row["member_order_sha256"] = "broken"
            with gzip.open(path, "wt") as handle:
                handle.write(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                audit_consumed(path, params, 1)

    def test_factorial_validity_fails_on_each_matched_invariant(self):
        rows = [{"source": "ukr_rus", "seed": 0, "policy": policy,
                 "initial_sha256": "init", "anchor_hashes": ["a", "b"],
                 "final_walk_rng_sha256": "walk", "member_set_hashes": [policy.split("_")[0]]}
                for policy in POLICIES]
        self.assertEqual(check_groups(rows), 1)
        for key in ("initial_sha256", "anchor_hashes", "final_walk_rng_sha256", "member_set_hashes"):
            changed = copy.deepcopy(rows)
            changed[1][key] = "broken"
            with self.assertRaises(ValueError):
                check_groups(changed)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            check_groups(rows[:-1])

    def test_weights_nonfinite_rejected(self):
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            model_digest({"weight": torch.tensor([float("nan")])})
        self.assertNotEqual(model_digest({"weight": torch.ones(3)}), model_digest({"weight": torch.zeros(3)}))


if __name__ == "__main__":
    unittest.main()
