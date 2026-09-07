import unittest

import torch
from torch_geometric.data import Data

from experiments.sampler import NeighborSampler
from models.gnn_with_edge_attr import PinSAGEConv


class PinSAGETest(unittest.TestCase):
    def test_sampler_returns_distinct_weighted_tokens(self):
        # Repeated walk visits become one token with one normalized weight.
        graph = Data(edge_index=torch.tensor([[0], [1]]), num_nodes=2)
        sampler = NeighborSampler(
            graph, num_hops=2, method="pinsage",
            pinsage_num_walks=8, pinsage_walk_length=2, pinsage_topk=2,
        )
        nodes, edges, edge_ids, weights = sampler.sample_node(0)
        self.assertEqual(nodes.tolist(), [0, 1])
        self.assertEqual(edges.tolist(), [[1], [0]])
        self.assertEqual(edge_ids.tolist(), [-1])
        self.assertTrue(torch.allclose(weights, torch.tensor([1.0])))

    def test_weighted_pooling_uses_visit_importance(self):
        layer = PinSAGEConv(1, None, 1, batch_norm=False)
        with torch.no_grad():
            layer.lin_x.weight.fill_(1)
            layer.lin_x.bias.zero_()
            layer.lin_self_loops.weight.zero_()
            layer.lin_self_loops.bias.zero_()
            layer.mlp = torch.nn.Identity()
        x = torch.tensor([[0.0], [2.0], [8.0]])
        edges = torch.tensor([[1, 2], [0, 0]])
        out = layer(x, edges, edge_weight=torch.tensor([0.75, 0.25]))
        self.assertAlmostEqual(out[0, 0].item(), 3.5)


if __name__ == "__main__":
    unittest.main()
