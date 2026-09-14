import torch

from mixture_scaling.precomputed_context import cache_key, tensor_sha256


def test_context_cache_key_changes_with_topology_and_scope():
    edges = torch.tensor([[0, 1], [1, 2]])
    base, receipt = cache_key("g", edges, 15, 0, 1024, "full")
    changed, _ = cache_key("g", edges.flip(0), 15, 0, 1024, "full")
    lp, _ = cache_key("g", edges, 15, 0, 1024, "lp_background")
    assert base != changed
    assert base != lp
    assert receipt["topology_sha256"] == tensor_sha256(edges)


def test_tensor_hash_is_stable_and_content_sensitive():
    value = torch.arange(12).reshape(2, 6)
    assert tensor_sha256(value) == tensor_sha256(value.clone())
    changed = value.clone(); changed[0, 0] = 99
    assert tensor_sha256(value) != tensor_sha256(changed)
