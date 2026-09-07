"""Tiny CPU fixtures for KG topology and Q/K/V intervention boundaries."""
import torch
from torch_geometric.data import Batch, Data

from .role_interventions import context_donor, transplant_projection


def test_donor_preserves_endpoints_features_and_query_edges():
    graphs = []
    for _ in range(2):
        graphs.append(Data(x=torch.arange(20).reshape(4, 5).float(),
                           edge_index=torch.tensor([[0, 2, 1], [2, 1, 2]]),
                           edge_attr=torch.arange(6).reshape(3, 2).float(),
                           supernode=torch.tensor([3]),
                           edge_index_supernode=torch.tensor([[0, 1], [3, 3]]),
                           edge_index_from_supernode=torch.tensor([[3, 3], [0, 1]])))
    graph = Batch.from_data_list(graphs)
    arguments = (graph, torch.zeros(2, 5), torch.eye(2),
                 torch.tensor([[0, 0, 1, 1], [2, 3, 2, 3]]),
                 torch.zeros(4, 2), torch.tensor([False, False, True, True]))
    changed, receipt = context_donor(arguments)
    assert receipt["removed_edges"] == 3
    assert torch.equal(changed[0].edge_index, graph.edge_index[:, 3:])
    assert torch.equal(changed[0].edge_attr, graph.edge_attr[3:])
    for name in ("x", "ptr", "batch", "supernode", "edge_index_supernode", "edge_index_from_supernode"):
        assert torch.equal(changed[0][name], graph[name]), name
    assert graph.edge_index.shape[1] == 6
    for original, clone in zip(arguments[1:], changed[1:]):
        assert torch.equal(original, clone)
    opposite, _ = context_donor(arguments, "query")
    assert torch.equal(opposite[0].edge_index, graph.edge_index[:, :3])


def test_projection_only_changes_nominated_support_blocks():
    projection = torch.nn.Linear(3, 9)
    inputs = torch.randn(5, 3)
    native = projection(inputs).detach()
    donor = native + 10
    support = torch.tensor([True, False, True, False, False])
    for component, blocks in (("keys", (1,)), ("values", (2,)), ("joint", (1, 2))):
        with transplant_projection(projection, donor, support, component):
            actual = projection(inputs)
        expected = native.clone()
        for block in blocks:
            expected[support, block * 3:(block + 1) * 3] += 10
        assert torch.equal(actual, expected)
        assert torch.equal(projection(inputs), native)
    try:
        with transplant_projection(projection, donor, support, "keys"):
            raise RuntimeError("fixture failure")
    except RuntimeError:
        pass
    assert not projection._forward_hooks


if __name__ == "__main__":
    test_donor_preserves_endpoints_features_and_query_edges()
    test_projection_only_changes_nominated_support_blocks()
    print("Role intervention tests passed")
