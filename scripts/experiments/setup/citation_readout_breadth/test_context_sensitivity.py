import unittest
import torch
from torch_geometric.data import Data
from .context_sensitivity import shuffle_checked


class ContextSensitivityTest(unittest.TestCase):
    def batch(self):
        graph = Data(x=torch.arange(24, dtype=torch.float32).reshape(12, 2),
            edge_index=torch.tensor([[0,1,4,5],[1,2,5,6]]),
            global_node_ids=torch.tensor([0,1,2,-1,4,5,6,-1,8,9,10,-1]),
            ptr=torch.tensor([0,4,8,12]), batch=torch.arange(3).repeat_interleave(4),
            task_id_per_sample=torch.tensor([0,0,1]))
        return [graph, torch.ones(3,2), torch.eye(3), torch.ones(2,3), torch.ones(3,2), torch.zeros(3)]

    def test_shuffle_preserves_noncontext_and_multiset(self):
        original = self.batch()
        changed, audit = shuffle_checked(original, 710000)
        self.assertGreater(audit["changed_rows"], 0)
        self.assertEqual(audit["eligible_rows"], 6)
        torch.testing.assert_close(original[0].x, self.batch()[0].x)
        for idx in ([1,2,5,6], [9,10]):
            torch.testing.assert_close(original[0].x[idx].sort(dim=0).values,
                                       changed[0].x[idx].sort(dim=0).values)
        again, _ = shuffle_checked(original, 710000)
        torch.testing.assert_close(again[0].x, changed[0].x, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
