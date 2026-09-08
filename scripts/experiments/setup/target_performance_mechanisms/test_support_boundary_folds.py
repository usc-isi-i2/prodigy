import unittest
import torch
from torch_geometric.data import Data
from .support_boundary_folds import balanced_support_folds, hide_support_labels


class FoldTests(unittest.TestCase):
    def batch(self):
        y = torch.tensor([0, 0, 0, 1, 1, 1, 0, 1])
        q = torch.tensor([False] * 6 + [True] * 2)
        edges = torch.stack((torch.arange(8).repeat_interleave(2), torch.tensor([8, 9] * 8)))
        labels = torch.nn.functional.one_hot(y, 2).float()
        attrs = torch.stack((q.repeat_interleave(2).float(), (labels * 2 - 1).flatten()), 1)
        attrs[q.repeat_interleave(2), 1] = 0
        return (Data(x=torch.randn(8, 4), task_id_per_sample=torch.zeros(8, dtype=torch.long)),
                torch.randn(2, 4), labels, edges, attrs, q.repeat_interleave(2))

    def test_hidden_labels_cannot_reach_model(self):
        b = self.batch()
        held = balanced_support_folds(b)[0]
        changed = list(b)
        changed[2] = b[2].clone()
        changed[4] = b[4].clone()
        changed[2][held] = changed[2][held].flip(1)
        changed[4][held.repeat_interleave(2) * 2 + torch.tensor([0, 1]).repeat(len(held)), 1] *= -1
        a, c = hide_support_labels(b, held), hide_support_labels(changed, held)
        for i in range(1, 6):
            self.assertTrue(torch.equal(a[i], c[i]))
        self.assertEqual(int(b[5].sum()), 4)
        self.assertEqual(int(a[5].sum()), 8)
        self.assertTrue(torch.equal(a[0].x, b[0].x))

    def test_folds_cover_support_once(self):
        b = self.batch()
        f = balanced_support_folds(b)
        self.assertEqual(tuple(f.shape), (3, 2))
        self.assertEqual(sorted(f.flatten().tolist()), list(range(6)))
        with self.assertRaises(ValueError):
            hide_support_labels(b, [6])


if __name__ == "__main__":
    unittest.main()
