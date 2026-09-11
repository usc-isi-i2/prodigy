import torch
from mixture_scaling.node_only_transfer import DirectLinkLoader
from mixture_scaling.prefetch_links import PrefetchLinks


def loaders():
    edges = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])
    return {s: DirectLinkLoader(torch.arange(48).reshape(12, 4).float() + i,
             edges, 2, 5, True, 7+i) for i, s in enumerate(("a", "b"))}


def test_prefetch_preserves_batches_across_epochs_and_partial_batches():
    schedule = ["a", "b"] * 8
    reference = loaders()
    iterators = {s: iter(v) for s, v in reference.items()}
    with PrefetchLinks(loaders(), schedule, "cpu", depth=5, workers=3) as fast:
        for source in schedule:
            try:
                expected = next(iterators[source])
            except StopIteration:
                iterators[source] = iter(reference[source])
                expected = next(iterators[source])
            actual_source, actual = next(fast)
            assert actual_source == source
            for field in ("x", "edge_label_index", "edge_label"):
                assert torch.equal(getattr(actual, field), getattr(expected, field))
        assert list(fast) == []


def test_prefetch_can_stop_early_without_dangling_threads():
    with PrefetchLinks(loaders(), ["a", "b"]*20, "cpu", depth=4, workers=2) as fast:
        next(fast)
    assert not fast.pending and fast.ready is None
