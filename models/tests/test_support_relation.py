import torch

from models.general_gnn import SingleLayerGeneralGNN


def make_model():
    return SingleLayerGeneralGNN(
        [], params={
            "emb_dim": 4, "task_name": "classification",
            "support_label_prototypes": True, "learned_relation_scorer": True,
        }
    )


def test_support_prototypes_ignore_queries_and_start_as_identity():
    model = make_model()
    inputs = torch.tensor([[1., 0, 0, 0], [9., 9, 9, 9], [0, 1., 0, 0]])
    labels = torch.randn(2, 4)
    # Complete bipartite edges; only edges 0 and 4 are positive supports.
    edge_index = torch.tensor([[0, 0, 1, 1, 2, 2], [3, 4, 3, 4, 3, 4]])
    edge_attr = torch.tensor([[0, 1], [0, -1], [1, 0], [1, 0], [0, -1], [0, 1.]])
    query = edge_attr[:, 0].bool()
    out = model.add_support_prototypes(inputs, labels, edge_index, edge_attr, query)
    torch.testing.assert_close(out, labels)
    out.sum().backward()
    assert model.prototype_adapter[-1].weight.grad is not None


def test_relation_residual_starts_as_cosine_and_gets_gradients():
    model = make_model()
    inputs = torch.randn(3, 4)
    labels = torch.randn(2, 4)
    edges = torch.tensor([[0, 0, 1, 1, 2, 2], [3, 4, 3, 4, 3, 4]])
    actual = model.decode(inputs, labels, edges)
    joined = torch.cat([inputs, labels])
    expected = model.cos(joined[edges[0]], joined[edges[1]]) * model.logit_scale.exp()
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert model.relation_scorer[-1].weight.grad is not None
