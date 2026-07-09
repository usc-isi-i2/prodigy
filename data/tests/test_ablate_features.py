"""Unit tests for the feature-ablation augmentation (data/augment.py).

These are pure-torch and need no cluster data, so they document + guard the
intervention used by the feature-ablation experiment
(scripts/experiments/feature_ablation/).

Run: python -m pytest data/tests/test_ablate_features.py
"""
import torch
from torch_geometric.data import Data

from data.augment import AblateAllFeatures, AblateEdges, get_aug


def _toy_graph():
    x = torch.arange(12, dtype=torch.float).reshape(4, 3)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    return Data(x=x, edge_index=edge_index)


def _toy_bidir_graph():
    """Mimics the loader's subgraph convention: 3 forward edges mirrored to 6,
    edge_attr a [E,1] direction flag (0=forward, 1=reverse), plus a trailing
    pooling-supernode node (index 4) whose edges live OUTSIDE edge_index."""
    x = torch.arange(15, dtype=torch.float).reshape(5, 3)  # node 4 = supernode
    fwd = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    edge_index = torch.cat([fwd, fwd.flip(0)], dim=1)
    edge_attr = torch.cat([torch.zeros(3), torch.ones(3)]).unsqueeze(1)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=5)


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


def test_rewire_preserves_nodes_features_and_edge_count():
    g = _toy_bidir_graph()
    torch.manual_seed(0)
    out = AblateEdges("rewire")(g)
    # node features, node count, and edge count are all preserved
    assert torch.equal(out.x, g.x)
    assert out.num_nodes == g.num_nodes
    assert out.edge_index.size(1) == g.edge_index.size(1)
    # edge_attr keeps the loader's [E,1] direction flag (first half 0, second 1)
    assert out.edge_attr.shape == g.edge_attr.shape
    assert torch.equal(out.edge_attr[:3, 0], torch.zeros(3))
    assert torch.equal(out.edge_attr[3:, 0], torch.ones(3))


def test_rewire_changes_structure_but_stays_in_support():
    g = _toy_bidir_graph()
    torch.manual_seed(1)
    out = AblateEdges("rewire")(g)
    # the adjacency is scrambled ...
    assert not torch.equal(out.edge_index, g.edge_index)
    # ... but never introduces the supernode (idx 4) or any out-of-support node:
    # real edges only touch {0,1,2,3}, so the rewire must too.
    assert set(out.edge_index.unique().tolist()).issubset({0, 1, 2, 3})
    # forward/reverse halves stay mirror images (bidirectional convention kept)
    m = out.edge_index.size(1) // 2
    assert torch.equal(out.edge_index[:, m:], out.edge_index[:, :m].flip(0))


def test_rewire_is_deterministic_under_seed():
    torch.manual_seed(7)
    a = AblateEdges("rewire")(_toy_bidir_graph()).edge_index
    torch.manual_seed(7)
    b = AblateEdges("rewire")(_toy_bidir_graph()).edge_index
    assert torch.equal(a, b)


def test_rewire_noop_on_edgeless_graph():
    g = Data(x=torch.zeros(3, 2), edge_index=torch.empty(2, 0, dtype=torch.long))
    out = AblateEdges("rewire")(g)
    assert out.edge_index.numel() == 0


def test_get_aug_ER_token_and_composition_with_feature_noise():
    g = _toy_bidir_graph()
    torch.manual_seed(0)
    out = get_aug("ER")(_toy_bidir_graph())
    assert out.edge_index.size(1) == g.edge_index.size(1)
    # the 2x2 "both" cell: feature noise (NR1.0) composed with edge rewire (ER)
    torch.manual_seed(0)
    both = get_aug("NR1.0,ER", node_feature_distribution=g.x)(_toy_bidir_graph())
    assert not torch.equal(both.x, g.x)             # features resampled
    assert not torch.equal(both.edge_index, g.edge_index)  # edges rewired


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok: {name}")
    print("all passed")
