import torch

from data.dataloader import KGCollator


def test_adapter_omits_endpoint_flags():
    collator = KGCollator(torch.zeros(3, 768), add_endpoint_flags=False)
    graph = type("Graph", (), {})()
    graph.x = torch.zeros(4, 768)
    task = {7: [graph, graph]}
    params = type("Params", (), {"n_aug": 1, "n_shot": 1})()
    graphs, _, _, _ = collator.process_one_task(task, params)
    assert graphs[0].x.shape == (4, 768)


def test_native_collator_adds_endpoint_flags():
    collator = KGCollator(torch.zeros(3, 768), add_endpoint_flags=True)
    graph = type("Graph", (), {})()
    graph.x = torch.zeros(4, 768)
    task = {7: [graph]}
    params = type("Params", (), {"n_aug": 1, "n_shot": 1})()
    graphs, _, _, _ = collator.process_one_task(task, params)
    assert graphs[0].x.shape == (4, 770)
    assert graphs[0].x[0, -1] == 1
    assert graphs[0].x[1, -2] == 1


def test_adapter_replaces_last_text_dims_with_endpoint_flags():
    collator = KGCollator(
        torch.zeros(3, 768),
        add_endpoint_flags=False,
        replace_text_dims_with_endpoint_flags=True,
    )
    graph = type("Graph", (), {})()
    graph.x = torch.arange(4 * 768, dtype=torch.float32).reshape(4, 768)
    original_prefix = graph.x[:, :-2].clone()
    task = {7: [graph]}
    params = type("Params", (), {"n_aug": 1, "n_shot": 1})()
    graphs, _, _, _ = collator.process_one_task(task, params)
    assert graphs[0].x.shape == (4, 768)
    assert torch.equal(graphs[0].x[:, :-2], original_prefix)
    assert graphs[0].x[0, -1] == 1
    assert graphs[0].x[1, -2] == 1
    assert graphs[0].x[2:, -2:].sum() == 0
