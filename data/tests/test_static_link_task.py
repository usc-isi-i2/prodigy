"""Unit test for data.midterm.StaticLinkTask sampling logic.

Uses fake neighbor samplers (duck-typed ``.whole_adj.csr()``) so it does not need
a real graph, but importing ``data.midterm`` pulls in torch_sparse — run in the
``prodigy`` env:

    python -m data.tests.test_static_link_task     # from repo root
"""

from __future__ import annotations

import random

import torch

from data.midterm import StaticLinkTask


class _FakeAdj:
    def __init__(self, rowptr, col):
        self._rowptr = torch.tensor(rowptr, dtype=torch.long)
        self._col = torch.tensor(col, dtype=torch.long)

    def csr(self):
        return self._rowptr, self._col, torch.arange(self._col.numel())


class _FakeSampler:
    """Build a CSR from an undirected edge list over ``size`` nodes."""

    def __init__(self, size, undirected_edges):
        adj = {i: [] for i in range(size)}
        for u, v in undirected_edges:
            adj[u].append(v)
            adj[v].append(u)
        rowptr = [0]
        col = []
        for i in range(size):
            col.extend(sorted(set(adj[i])))
            rowptr.append(len(col))
        self.whole_adj = _FakeAdj(rowptr, col)


def test_positives_from_holdout_and_hard_negatives_from_two_hop():
    size = 8
    # Background chain 0-1-2-3-4-5-6-7 -> node 0's 2-hop (not 1-hop) = {2}.
    background = _FakeSampler(size, [(i, i + 1) for i in range(size - 1)])
    # Held-out edges incident to center 0: 0-5, 0-6 (two positives available).
    holdout = _FakeSampler(size, [(0, 5), (0, 6)])

    task = StaticLinkTask(holdout, background, size, neg_ratio=1, hard_negatives=True)
    rng = random.Random(0)
    episode = task.sample(num_label=2, num_member=2, num_shot=1, num_query=1, rng=rng)

    keys = list(episode.keys())
    # (0, center)=negatives first, (1, center)=positives second.
    assert keys[0][0] == 0 and keys[1][0] == 1
    center = keys[0][1]
    assert center == 0
    positives = set(episode[(1, 0)])
    negatives = episode[(0, 0)]

    assert positives <= {5, 6}                       # positives are held-out neighbors
    assert 0 not in negatives                         # never the center
    assert positives.isdisjoint(set(negatives))       # neg/pos disjoint
    assert len(negatives) == 2                         # neg_ratio 1 * num_member 2
    # Node 0's 2-hop-not-1-hop shell in the background chain is exactly {2}; it must
    # be picked as a hard negative before random fallback tops up the rest.
    assert 2 in negatives
    print("ok: static LP positives from holdout, hard negatives from 2-hop shell")


def test_falls_back_when_no_center_has_enough_positives():
    size = 5
    background = _FakeSampler(size, [(0, 1), (1, 2)])
    holdout = _FakeSampler(size, [(0, 1)])  # only 1 positive anywhere
    task = StaticLinkTask(holdout, background, size, neg_ratio=1, hard_negatives=True)
    rng = random.Random(1)
    try:
        task.sample(2, 3, 1, 2, rng)  # need >=3 positives -> impossible
    except RuntimeError as exc:
        assert "held-out" in str(exc)
        print("ok: static LP raises when no center has enough positives")
        return
    raise AssertionError("expected RuntimeError for insufficient positives")


if __name__ == "__main__":
    test_positives_from_holdout_and_hard_negatives_from_two_hop()
    test_falls_back_when_no_center_has_enough_positives()
    print("\nAll static_link_task tests passed.")
