import unittest
import copy
import torch

from models.nm_geometry_readout import GeometryResidualReadout, mean_support_cosine
from .test_metagraph import fixture
from models.layer_classes import SupernodeAggrLayer
from torch_geometric.data import Data


class GeometryReadoutTest(unittest.TestCase):
    def test_production_forward_opt_in_and_no_query_label_leak(self):
        class Pool(torch.nn.Module, SupernodeAggrLayer):
            def forward(self, x, edges, index, batch):
                return x[index]
        model, (pre, labels, edges, attrs, roles) = fixture()
        model.layer_list = torch.nn.ModuleList([Pool(), model.layer_list[2]])
        model.params.update(ignore_label_embeddings=False, zero_label_embeddings=False)
        n = len(pre)
        graph = Data(x=pre.clone(), supernode=torch.zeros(n, dtype=torch.long),
            ptr=torch.arange(n+1), batch=torch.arange(n),
            edge_index_supernode=torch.empty(2, 0, dtype=torch.long))
        truth = torch.nn.functional.one_hot(torch.arange(n)//3, 3).float()
        native = model(graph.clone(), labels, truth, edges, attrs, roles)[1]
        patched = copy.deepcopy(model)
        patched.nm_geometry_residual = True
        patched.geometry_readout = GeometryResidualReadout()
        self.assertEqual(set(patched.state_dict()), set(model.state_dict()))
        query = roles.reshape(n, 3)[:, 0]
        expected = GeometryResidualReadout()(native,
            mean_support_cosine(pre, edges, attrs, roles, 3)[query])
        actual = patched(graph.clone(), labels, truth, edges, attrs, roles)[1]
        torch.testing.assert_close(actual, expected)
        changed = truth.roll(1, dims=1)
        torch.testing.assert_close(actual, patched(graph.clone(), labels, changed, edges, attrs, roles)[1])
        patched.train()
        with self.assertRaises(ValueError):
            patched(graph.clone(), labels, truth, edges, attrs, roles)

    def test_observed_support_only_and_class_permutation(self):
        _, (pre, _, edges, attrs, roles) = fixture()
        result = mean_support_cosine(pre, edges, attrs, roles, 3)
        expected = torch.stack([torch.nn.functional.cosine_similarity(pre[:, None],
            pre[k * 3:k * 3 + 2][None], dim=2).mean(1) for k in range(3)], 1)
        torch.testing.assert_close(result, expected)
        # Even adversarial query signs must not leak query labels into prototypes.
        changed = attrs.clone()
        changed[roles, 1] = 1
        torch.testing.assert_close(result, mean_support_cosine(pre, edges, changed, roles, 3))
        perm = torch.tensor([2, 0, 1])
        order = torch.arange(len(roles)).reshape(-1, 3)[:, perm].flatten()
        permuted = mean_support_cosine(pre, edges[:, order], attrs[order], roles[order], 3)
        torch.testing.assert_close(permuted, result[:, perm])

    def test_disconnected_episodes_and_basis_invariance(self):
        _, (pre, _, edges, attrs, roles) = fixture()
        units, _ = torch.linalg.qr(torch.randn(pre.shape[1], pre.shape[1]))
        torch.testing.assert_close(mean_support_cosine(pre @ units, edges, attrs, roles, 3),
            mean_support_cosine(pre, edges, attrs, roles, 3), atol=1e-6, rtol=1e-5)
        n = len(pre)
        other = pre * torch.linspace(.5, 2, n)[:, None]
        src = torch.cat([edges[0], edges[0] + n])
        dst = torch.cat([edges[1] + n, edges[1] + n + 3])
        both = mean_support_cosine(torch.cat([pre, other]), torch.stack([src, dst]),
            attrs.repeat(2, 1), roles.repeat(2), 3)
        torch.testing.assert_close(both[:n], both[n:], atol=1e-6, rtol=1e-5)

    def test_scale_invariance_finite_ties_and_symmetry(self):
        head = GeometryResidualReadout()
        a = torch.tensor([[1., 2., 4.], [3., 3., 3.]])
        b = torch.tensor([[.3, .1, .2], [.5, .5, .5]])
        torch.testing.assert_close(head(a, b), head(a * 7 + 20, b * 3 - 1))
        torch.testing.assert_close(head(a, b), head(b, a))
        self.assertTrue(torch.isfinite(head(a, b)).all())
        self.assertTrue(torch.equal(head(a, b)[1], torch.zeros(3)))
        self.assertEqual(list(head.parameters()), [])


if __name__ == '__main__':
    unittest.main()
