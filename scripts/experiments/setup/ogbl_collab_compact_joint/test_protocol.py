import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import torch

from run import Model, COUNTS, comparison, gate, numeric_count, sampled_indices, adjacency, shared


class CompactJointTests(unittest.TestCase):
    def test_parameter_counts_and_no_node_table(self):
        for arm in COUNTS:
            model = Model(arm)
            self.assertEqual(sum(p.numel() for p in model.parameters()), COUNTS[arm])
            self.assertFalse(any(isinstance(m, torch.nn.Embedding) for m in model.modules()))

    def test_symmetric_decoder_and_self_rule(self):
        torch.manual_seed(5)
        for arm in COUNTS:
            model = Model(arm)
            z = torch.randn(5, 256) if arm == 'joint' else None
            edges = torch.tensor([[0, 1], [3, 2], [4, 4]])
            features = torch.randn(3, 14)
            a = model.score(z, edges, features)
            b = model.score(z, edges.flip(1), features)
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertEqual(a[-1].item(), -1e9)

    def test_graph_encoder_receives_gradients(self):
        torch.manual_seed(2)
        model = Model('joint')
        x = torch.randn(5, 128)
        adj = adjacency(np.array([[0, 1], [1, 2], [2, 3]]), 5, 'cpu')
        z = model.encode(x, adj)
        score = model.score(z, torch.tensor([[0, 2], [1, 3]]), torch.randn(2, 14))
        score.sum().backward()
        self.assertGreater(sum(p.grad.abs().sum().item() for p in model.encoder.parameters()), 0)

    def test_uniform_stream_independent_of_initialization_and_mined_pool(self):
        a = torch.Generator().manual_seed(0)
        b = torch.Generator().manual_seed(0)
        for _ in range(5):
            pa, na, ba = sampled_indices(a, 3000, 5000, torch.arange(2048))
            torch.randn(1000)
            pb, nb, bb = sampled_indices(b, 3000, 5000, torch.arange(2048)+2048)
            self.assertEqual(ba, bb)
            self.assertTrue(torch.equal(pa, pb))
            self.assertTrue(torch.equal(na[:1024], nb[:1024]))
            self.assertFalse(torch.equal(na[1024:], nb[1024:]))

    def test_recomputed_cutoff_can_lose_unchanged_positives(self):
        bp = np.array([1.5, 2.5, 3.5])
        bn = np.ones(100)
        neg = np.r_[np.full(50, 3.), np.zeros(50)]
        features = np.zeros((3, 13)); features[0, 8] = 1
        result = comparison(bp.copy(), neg, bp, bn, {'pfeatures': features, 'bp': bp})
        self.assertEqual(result['lost'], 2)
        self.assertEqual(result['net_hits'], -2)

    def test_auxiliary_scalar_count(self):
        self.assertEqual(numeric_count({'a': 3., 'b': [1, 2], 'mode': 'x', 'flag': True}), 3)

    def test_symmetric_feature_path_preserves_legacy_and_reverses_exactly(self):
        n = 20
        train = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [0, 3], [1, 4]])
        years = np.array([2014]*4+[2016]*2)
        graph = {'num_nodes': n, 'edge_index': train.T, 'edge_year': years,
                 'edge_weight': np.ones(len(train)), 'node_feat': np.eye(n, dtype=np.float32)}
        split = {'train': {'edge': train, 'year': years}}
        neg = np.array([(i, j) for i in range(5, n) for j in range(4)])

        def raw(edges, **kwargs):
            value = ((edges[:, 0]+1)/(edges[:, 1]+1)).astype(np.float32)
            return (value, value, None, edges[:, 0].astype(np.float32)/n,
                    {'l3_values': np.full(len(edges), np.nan, dtype=np.float32)})

        aa = SimpleNamespace(build_weighted_adj=lambda *a: None,
             precompute_inv_log_deg=lambda *a: None, build_adj=lambda *a: [[0] for _ in range(n)],
             compute_exact_lcc=lambda *a: np.linspace(0, 1, n), precompute_aa_matrix=lambda *a: None,
             score_edges=raw, compute_gate_value=lambda *a: 1.)
        calibration = {'gate': {}, 'anchor_scale': 1.}
        with patch.object(gate, 'negatives', return_value=neg):
            legacy, _, _ = gate.make_year(aa, shared, graph, split, 2016, calibration)
            new, _, _ = gate.make_year(aa, shared, graph, split, 2016, calibration, symmetric_features=True)
        for key in legacy:
            np.testing.assert_array_equal(legacy[key], new[key])
        reversed_train = train.copy()
        reversed_train[years == 2016] = reversed_train[years == 2016, ::-1]
        with patch.object(gate, 'negatives', return_value=neg[:, ::-1].copy()):
            reverse, _, _ = gate.make_year(aa, shared, graph,
                {'train': {'edge': reversed_train, 'year': years}}, 2016, calibration, symmetric_features=True)
        np.testing.assert_array_equal(new['psymmetric'], reverse['psymmetric'])
        np.testing.assert_array_equal(new['nsymmetric'], reverse['nsymmetric'])

    def test_historical_inputs_exclude_future_weights(self):
        graph = {'edge_index': np.array([[0, 1, 0, 1], [1, 0, 1, 0]]),
                 'edge_year': np.array([2014, 2014, 2017, 2017]),
                 'edge_weight': np.array([2., 2., 999., 999.])}
        split = {'train': {'edge': np.array([[0, 1], [0, 1]]), 'year': np.array([2014, 2017])}}
        _, years, weights, _ = gate.historical_inputs(graph, split, 2015)
        self.assertEqual(years.tolist(), [2014])
        np.testing.assert_allclose(weights, [1.9])


if __name__ == '__main__':
    unittest.main()
