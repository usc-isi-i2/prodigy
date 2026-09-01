import torch

from experiments.task_families import (
    TASK_FAMILY_TO_ID,
    effective_task_family,
    resolve_task_family,
)
from models.general_gnn import SingleLayerGeneralGNN


def params(dim=8, dropout=0.0, dataset="twibot20", seen="neighbor_matching", fusion="add"):
    return {
        "emb_dim": 16,
        "task_name": "classification",
        "dataset": dataset,
        "task_embedding_dim": dim,
        "task_embedding_dropout": dropout,
        "task_embedding_fusion": fusion,
        "task_embedding_seen_families": seen,
        "ignore_label_embeddings": False,
        "zero_label_embeddings": False,
        "zero_shot": False,
        "skip_path": False,
    }


def test_semantic_family_mapping_and_unknown_fallback():
    assert resolve_task_family("classification", "covid_political") == "political_leaning"
    assert resolve_task_family("classification", "election2020") == "political_leaning"
    assert resolve_task_family("neighbor_matching", "twibot20") == "neighbor_matching"
    assert effective_task_family("classification", "twibot20", "neighbor_matching") == "unknown"


def test_dimension_zero_is_identity():
    model = SingleLayerGeneralGNN(torch.nn.ModuleList(), params=params(dim=0))
    x, labels = torch.randn(3, 16), torch.randn(2, 16)
    out_x, out_labels = model.task_condition(object(), x, labels)
    assert out_x is x
    assert out_labels is labels


def test_conditioning_uses_batch_family_and_backpropagates():
    model = SingleLayerGeneralGNN(torch.nn.ModuleList(), params=params())
    graph = type("Graph", (), {})()
    graph.task_family_id = torch.tensor(TASK_FAMILY_TO_ID["political_leaning"])
    x, labels = torch.zeros(3, 16), torch.zeros(2, 16)
    out_x, out_labels = model.task_condition(graph, x, labels)
    (out_x.sum() + out_labels.sum()).backward()
    assert model.task_embedding.weight.grad[TASK_FAMILY_TO_ID["political_leaning"]].abs().sum() > 0


def test_dropout_one_replaces_family_with_unknown():
    model = SingleLayerGeneralGNN(torch.nn.ModuleList(), params=params(dropout=1.0))
    model.train()
    graph = type("Graph", (), {})()
    graph.task_family_id = torch.tensor(TASK_FAMILY_TO_ID["political_leaning"])
    x, labels = torch.zeros(1, 16), torch.zeros(1, 16)
    out_x, _ = model.task_condition(graph, x, labels)
    expected = model.task_to_input(model.task_embedding.weight[TASK_FAMILY_TO_ID["unknown"]])
    assert torch.allclose(out_x[0], expected)


def test_film_starts_as_exact_identity_and_learns_projection():
    model = SingleLayerGeneralGNN(torch.nn.ModuleList(), params=params(fusion="film"))
    graph = type("Graph", (), {})()
    graph.task_family_id = torch.tensor(TASK_FAMILY_TO_ID["political_leaning"])
    x, labels = torch.randn(3, 16), torch.randn(2, 16)
    out_x, out_labels = model.task_condition(graph, x, labels)
    assert torch.allclose(out_x, x)
    assert torch.allclose(out_labels, labels)
    (out_x.sum() + out_labels.sum()).backward()
    assert model.task_input_beta.weight.grad.abs().sum() > 0
    assert model.task_label_beta.weight.grad.abs().sum() > 0
