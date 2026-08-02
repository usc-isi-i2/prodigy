from __future__ import annotations

import unittest

import torch

from models.gnn_with_edge_attr import GATv2ConvOptionalEdgeAttr


class GATv2OptionalEdgeAttrTest(unittest.TestCase):
    def setUp(self) -> None:
        self.x = torch.randn(4, 8)
        self.edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
        self.edge_attr = torch.randn(self.edge_index.shape[1], 1)

    def test_ignores_present_edge_attr_when_edge_dim_is_disabled(self) -> None:
        conv = GATv2ConvOptionalEdgeAttr(8, 4, edge_dim=None)
        with_attrs = conv(self.x, self.edge_index, edge_attr=self.edge_attr)
        without_attrs = conv(self.x, self.edge_index, edge_attr=None)
        torch.testing.assert_close(with_attrs, without_attrs)

    def test_preserves_edge_attr_when_edge_dim_is_enabled(self) -> None:
        conv = GATv2ConvOptionalEdgeAttr(8, 4, edge_dim=1)
        output = conv(self.x, self.edge_index, edge_attr=self.edge_attr)
        self.assertEqual(tuple(output.shape), (4, 4))


if __name__ == "__main__":
    unittest.main()
