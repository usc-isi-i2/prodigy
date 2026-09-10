import torch

from mixture_scaling.context_views import fixed_view, neighbor_mean, view_dim


def test_neighbor_mean_excludes_self_loops_and_uses_incoming_edges():
    x = torch.tensor([[1.0], [3.0], [9.0]])
    edges = torch.tensor([[0, 1, 2, 1], [1, 0, 0, 1]])
    assert torch.equal(neighbor_mean(x, edges), torch.tensor([[6.0], [1.0], [0.0]]))


def test_fixed_views_have_expected_content_and_width():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    edges = torch.tensor([[0, 1], [1, 0]])
    neighborhood = fixed_view(x, edges, "neighborhood")
    combined = fixed_view(x, edges, "node_neighborhood")
    assert torch.equal(neighborhood, x.flip(0))
    assert torch.equal(combined, torch.cat((x, x.flip(0)), dim=1))
    assert view_dim(2, "neighborhood") == 2
    assert view_dim(2, "node_neighborhood") == 4
