import torch

from models.general_gnn import SingleLayerGeneralGNN


def model(threshold=0.8, alpha=1.0, iterations=1):
    return SingleLayerGeneralGNN([], params={
        "emb_dim": 2, "task_name": "classification",
        "transductive_threshold": threshold,
        "transductive_alpha": alpha,
        "transductive_iterations": iterations,
    })


def complete_edges(rows, classes):
    src = torch.arange(rows).repeat_interleave(classes)
    dst = torch.arange(classes).repeat(rows) + rows
    return torch.stack([src, dst])


def test_confident_queries_refine_only_their_predicted_class():
    m = model(threshold=0.7)
    inputs = torch.tensor([[1., 0.], [0., 1.], [1., 0.], [.9, .1]])
    labels = torch.tensor([[1., 0.], [0., 1.]])
    edges = complete_edges(4, 2)
    query_edges = torch.tensor([False] * 4 + [True] * 4)
    refined = m.refine_labels_transductively(inputs, labels, edges, query_edges, 2)
    torch.testing.assert_close(refined[0], torch.tensor([.95, .05]))
    torch.testing.assert_close(refined[1], labels[1])


def test_high_threshold_is_noop_and_query_labels_are_not_inputs():
    m = model(threshold=1.1)
    inputs = torch.randn(4, 2)
    labels = torch.randn(2, 2)
    edges = complete_edges(4, 2)
    query_edges = torch.tensor([False] * 4 + [True] * 4)
    refined = m.refine_labels_transductively(inputs, labels, edges, query_edges, 2)
    torch.testing.assert_close(refined, labels)
