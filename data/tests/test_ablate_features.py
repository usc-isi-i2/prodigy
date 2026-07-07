"""Unit tests for the feature-ablation augmentation (data/augment.py).

These are pure-torch and need no cluster data, so they document + guard the
intervention used by the feature-ablation experiment
(scripts/experiments/feature_ablation/).

Run: python -m pytest data/tests/test_ablate_features.py
"""
import torch
from torch_geometric.data import Data

from data.augment import AblateAllFeatures, get_aug


def _toy_graph():
    x = torch.arange(12, dtype=torch.float).reshape(4, 3)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    return Data(x=x, edge_index=edge_index)


def test_zero_ablation_zeroes_all_features():
    g = _toy_graph()
    out = AblateAllFeatures("zero")(g)
    assert torch.equal(out.x, torch.zeros_like(g.x))
    # original preserved for inspection, input graph untouched (copy semantics)
    assert torch.equal(out.x_orig, g.x)


def test_permute_preserves_multiset_but_breaks_alignment():
    g = _toy_graph()
    torch.manual_seed(0)
    out = AblateAllFeatures("permute")(g)
    # same set of feature rows, but at least one row moved
    assert torch.equal(out.x.sum(0), g.x.sum(0))
    assert not torch.equal(out.x, g.x)


def test_permute_is_deterministic_under_seed():
    g = _toy_graph()
    torch.manual_seed(42)
    a = AblateAllFeatures("permute")(_toy_graph()).x
    torch.manual_seed(42)
    b = AblateAllFeatures("permute")(_toy_graph()).x
    assert torch.equal(a, b)


def test_get_aug_tokens():
    g = _toy_graph()
    assert torch.equal(get_aug("FZ")(_toy_graph()).x, torch.zeros_like(g.x))
    torch.manual_seed(0)
    permuted = get_aug("FP")(_toy_graph()).x
    assert torch.equal(permuted.sum(0), g.x.sum(0))


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok: {name}")
    print("all passed")
