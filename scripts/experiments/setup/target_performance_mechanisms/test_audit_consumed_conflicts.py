import copy
import gzip
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from .audit_consumed_conflicts import audit_stream, digest, episode_stats, normalized


class ConflictTests(unittest.TestCase):
    def test_no_conflicts_and_context_includes_center(self):
        s, cases = episode_stats([[0, 1, 2], [3, 4, 5]], [0, 1, 1], [[1, 2, 3], [4, 1, 5]])
        n = normalized(s)
        self.assertEqual(n["identity_only_query_accuracy_upper_bound"], 1)
        self.assertEqual(n["identity_only_query_ce_lower_bound"], 0)
        self.assertEqual(n["query_without_context_fraction"], .25)
        self.assertEqual(cases, {})

    def test_cross_role_identity_always_wrong_class(self):
        s, c = episode_stats([[10, 11, 12], [11, 10, 13]], [0, 1, 1], np.ones((2, 3)))
        self.assertEqual(s["query_also_wrong_class_support_positions"], 2)
        self.assertEqual(s["query_multilabel_query_positions"], 0)
        self.assertEqual(c["query_support"]["center_node_id"], 10)
        self.assertEqual([v["role"] for v in c["query_support"]["occurrences"]], ["support", "query"])

    def test_identical_queries_have_an_identity_only_bound(self):
        s, _ = episode_stats([[0, 9, 8], [1, 9, 7], [2, 9, 8]], [0, 1, 1], np.ones((3, 3)))
        n = normalized(s)
        self.assertEqual(s["query_multilabel_query_positions"], 5)
        self.assertEqual(n["identity_only_query_accuracy_upper_bound"], 3/6)
        self.assertAlmostEqual(n["identity_only_query_ce_lower_bound"], (3*math.log(3)+2*math.log(2))/6)

    def test_role_swap_changes_cross_role_conflict_without_identity_change(self):
        a = [[0, 8, 9], [1, 8, 10]]
        s, _ = episode_stats(a, [0, 1, 1], np.ones((2, 3)))
        b = [[8, 0, 9], [1, 8, 10]]
        t, _ = episode_stats(b, [0, 1, 1], np.ones((2, 3)))
        self.assertEqual(s["duplicate_positions"], t["duplicate_positions"])
        self.assertEqual(s["query_also_wrong_class_support_positions"], 0)
        self.assertEqual(t["query_also_wrong_class_support_positions"], 1)

    def test_within_class_duplicates_rejected(self):
        with self.assertRaisesRegex(ValueError, "within-class"):
            episode_stats([[1, 1, 2]], [0, 1, 1], [[2, 2, 2]])

    def test_stream_hashes_summary_and_missing_step(self):
        ids = [[[0, 1, 2], [3, 1, 4]]]
        row = {"step": 1, "anchor_ids": [[10, 11]], "member_ids": ids,
            "query_roles": [0, 1, 1], "context_node_counts": [[[2]*3]*2], "source_ids": [0],
            "anchor_sha256": digest([[10, 11]]), "member_order_sha256": digest(ids),
            "member_set_sha256": digest(np.sort(ids, axis=-1))}
        params = {"batch_size": 1, "n_way": 2, "n_shots": 1, "n_query": 2}
        ref = {"steps": 1, "episodes": 1, "member_positions": 6, "unique_members": 5,
            "unique_anchors": 2, "effective_member_count": 36/8, "cross_class_duplicate_fraction": 1/6,
            "context_nodes_mean": 2., "source_ids": {"0": 1}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"records.gz"
            def write(r):
                with gzip.open(path, "wt") as f:
                    f.write(json.dumps(r)+"\n")
            write(row)
            result = audit_stream(path, params, ref, steps=1)
            self.assertEqual(result["summary"]["identity_only_query_accuracy_upper_bound"], .75)
            with self.assertRaisesRegex(ValueError, "incomplete"):
                audit_stream(path, params, ref, steps=2)
            changed = copy.deepcopy(ref)
            changed["unique_members"] = 4
            with self.assertRaisesRegex(ValueError, "summary mismatch"):
                audit_stream(path, params, changed, steps=1)
            changed = copy.deepcopy(row)
            changed["member_ids"][0][0][0] = 12
            write(changed)
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                audit_stream(path, params, ref, steps=1)


if __name__ == "__main__":
    unittest.main()
