import torch

from mixture_scaling.benchmark_node_only_throughput import endpoint_ids


def test_endpoint_ids_have_fixed_five_to_one_negative_ratio_and_replay():
    edges = torch.tensor([[0, 2, 4], [1, 3, 5]])
    first_ids, first_pairs = endpoint_ids(edges, 10, 8, torch.Generator().manual_seed(3))
    second_ids, second_pairs = endpoint_ids(edges, 10, 8, torch.Generator().manual_seed(3))
    assert first_pairs == second_pairs == 48
    assert first_ids.shape == (96,)
    assert torch.equal(first_ids, second_ids)
    src, dst = first_ids[:first_pairs], first_ids[first_pairs:]
    assert not torch.any(src[8:] == dst[8:])
