import unittest
from types import SimpleNamespace

import torch

from .replay import episode_probe
from .run_cross_model_matching import disagreement, matched_prototypes


class MatchingTests(unittest.TestCase):
    def batch(self):
        graph = SimpleNamespace(ptr=torch.arange(5), global_node_ids=torch.arange(4),
            task_id_per_sample=torch.zeros(4, dtype=torch.long), task_label_map=torch.tensor([[1, 0]]))
        batch = [graph, None, torch.tensor([[1., 0.], [0., 1.], [1., 0.], [0., 1.]]),
            None, None, torch.tensor([False, False, False, False, True, True, True, True])]
        return batch

    def test_matching_orientation_and_query_blind_prediction(self):
        batch = self.batch()
        x = torch.tensor([[1., 0.], [0., 1.], [1., 0.], [0., 1.]])
        y = torch.tensor([1, 0, 1, 0])
        result = matched_prototypes(x, batch, y)
        torch.testing.assert_close(result, torch.eye(2), rtol=0, atol=0)
        batch[2][2:] = batch[2][2:].flip(1)
        torch.testing.assert_close(episode_probe(x, batch, "prototype"), result, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            matched_prototypes(x, batch, y)

    def test_error_partition(self):
        a = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
        b = torch.tensor([[1., 0.], [0., 1.], [1., 0.], [0., 1.]])
        r = disagreement(a, b, torch.zeros(4, dtype=torch.long))
        self.assertEqual(r, {"queries": 4, "both_correct": 1, "both_wrong": 1,
            "prodigy_only_correct": 1, "samgpt_only_correct": 1})


if __name__ == "__main__":
    unittest.main()
