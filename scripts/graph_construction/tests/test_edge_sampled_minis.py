import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0, str(Path(__file__).parents[1]))
from edge_sampled_minis import sample_endpoints, nonself_isolates
from nonzero_feature_views import induced, verify_induced


class EdgeSamplingTest(unittest.TestCase):
    def test_exact_cap_no_isolates_and_reproducibility(self):
        # Self-loops cannot supply the promised connection to another node.
        edges = torch.tensor([[0, 0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6, 0]])
        a, witness, _ = sample_endpoints(edges, 7, 5, seed=2, batch_size=8, max_draws=1000)
        b, other, _ = sample_endpoints(edges, 7, 5, seed=2, batch_size=8, max_draws=1000)
        self.assertTrue(torch.equal(a, b) and torch.equal(witness, other))
        self.assertEqual(len(a), 5)
        g = {'x': torch.ones(7, 2), 'edge_index': edges}
        out = induced(g, a)
        verify_induced(g, out, a)
        self.assertEqual(nonself_isolates(out['edge_index'], 5), 0)

    def test_complete_induction_includes_unsampled_edges(self):
        edges = torch.tensor([[0, 1, 0], [1, 2, 2]])
        ids, witness, _ = sample_endpoints(edges, 3, 3, seed=0)
        out = induced({'x': torch.ones(3, 2), 'edge_index': edges}, ids)
        self.assertEqual(len(witness), 2)
        self.assertEqual(out['edge_index'].shape[1], 3)

    def test_impossible_nonself_selection_fails(self):
        with self.assertRaises(ValueError):
            sample_endpoints(torch.tensor([[0, 1], [0, 1]]), 2, 2, max_draws=10)

if __name__ == '__main__':
    unittest.main()
