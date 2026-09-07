import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from .run import check_identities, plan, probe, query_truth


class CitationReadoutTest(unittest.TestCase):
    def test_production_onehot_output_contract(self):
        labels = torch.eye(7).repeat(7, 1)
        query = torch.arange(49) >= 21
        returned = labels[query].clone()
        torch.testing.assert_close(query_truth(returned, labels, query), returned.argmax(1))
        with self.assertRaises(ValueError):
            query_truth(returned.argmax(1), labels, query)
        with self.assertRaises(ValueError):
            query_truth(returned.roll(1, 0), labels, query)

    def test_query_labels_never_used(self):
        torch.manual_seed(0)
        x = torch.randn(12, 8)
        query = torch.tensor([False, False, True, True] * 3)
        tasks = torch.tensor([0] * 4 + [1] * 4 + [2] * 4)
        labels = torch.eye(2).repeat(6, 1)
        corrupt = labels.clone()
        corrupt[query] = float("nan")
        for center in ("none", "support"):
            torch.testing.assert_close(probe(x, labels, query, tasks, center),
                                       probe(x, corrupt, query, tasks, center), rtol=0, atol=0)

    def test_task_local_support_centering(self):
        x = torch.randn(8, 5)
        labels = torch.eye(2).repeat(4, 1)
        query = torch.tensor([False, False, True, True] * 2)
        tasks = torch.tensor([0] * 4 + [1] * 4)
        a = probe(x, labels, query, tasks, "support")
        shifted = x.clone()
        shifted[4:] += 10
        b = probe(shifted, labels, query, tasks, "support")
        torch.testing.assert_close(a[:2], b[:2], rtol=0, atol=0)
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)

    def test_identity_guard(self):
        query = torch.tensor([False, False, True, True])
        tasks = torch.zeros(4, dtype=torch.long)
        check_identities(torch.arange(4), query, tasks)
        with self.assertRaises(ValueError):
            check_identities(torch.tensor([0, 1, 0, 3]), query, tasks)

    def test_frozen_plan(self):
        p = plan(SimpleNamespace(blocked=Path("/a"), interleaved=Path("/b"), graph=Path("/c")))
        self.assertEqual((p["ways"], p["shots"], p["queries_per_class"], p["episodes"]), (7, 3, 4, 128))
        self.assertEqual(p["episode_sampler_seed"], 400457)
        self.assertEqual(p["module_mode"], "eval")


if __name__ == "__main__":
    unittest.main()
