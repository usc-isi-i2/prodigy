"""Pair-conditioned link-prediction evaluator for frozen encoders.

Replaces the episodic ``static_link_prediction`` eval, which could not measure
link prediction at all. That path had three independent defects:

1. **Center-blind scoring.** ``StaticLinkTask.sample`` returns
   ``{(0, center): neg, (1, center): pos}`` (data/midterm.py). The ``center`` is
   only the second element of the label-map *key*; the subgraphs that get encoded
   are built from the candidate lists alone. So the score was ``f(v)`` -- the
   other endpoint of the queried edge never entered the model input.
2. **Frozen random class prototypes.** ``ignore_label_embeddings`` defaults True,
   so label reps are rows of ``nn.Embedding(1000, emb_dim)`` (general_gnn.py) and
   are frozen during pretraining (trainer.py). With ``--shots 0`` the eval also
   sets ``--zero_shot True``, which makes ``forward_metagraph`` skip message
   passing entirely -- so no support example could inform them either. The two
   "edge / no-edge" prototypes were therefore fixed random vectors, and the
   reported AUC measured alignment of a single node embedding with an arbitrary
   direction.
3. **Degree-confounded negatives.** Positives were drawn from a center's holdout
   neighbours (so every positive has holdout-degree >= 1); negatives had no such
   condition. A purely degree-sensitive encoder scored above chance.

Here every score is a function of BOTH endpoints, prototypes are not involved,
and negatives are degree-matched by construction. The evaluator is deliberately
standalone: it does not route through the metagraph/prototype machinery, because
that machinery is what was broken.

Protocol
--------
* Encode each node's k-hop subgraph on the ``static_background`` view, from which
  the held-out edges have already been removed -- the encoder never aggregates
  over an edge it is scored on.
* Score a pair with a symmetric function of the two endpoint embeddings
  (cosine or dot); no head is fitted, so this is a true zero-shot read.
* Compare against topology heuristics (common neighbours, Adamic-Adar,
  preferential attachment, Jaccard) and a raw-feature cosine floor, all computed
  on the same pair set.
* Lock score orientation on a validation split, then apply it to test, so a
  sign-flipped signal cannot masquerade as a below-chance result.

Run ``--self-test`` for the synthetic-graph gate (no checkpoint required).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# --------------------------------------------------------------------------- #
# Adjacency helpers (numpy/scipy only -- no torch needed, so the heuristics and
# the pair sampler are testable without a GPU or a checkpoint).
# --------------------------------------------------------------------------- #


@dataclass
class Adjacency:
    """Undirected CSR adjacency over ``num_nodes`` nodes.

    Built via scipy.sparse so it scales to the real retweet graphs (covid19 is
    23M nodes / 91M background edges); a numpy lexsort over 182M symmetrised
    pairs, or any per-node Python loop, is not viable at that size. Row indices
    are sorted, which ``contains_pairs`` and the heuristics rely on.
    """

    indptr: np.ndarray
    indices: np.ndarray
    num_nodes: int

    @classmethod
    def from_edge_index(cls, edge_index: np.ndarray, num_nodes: int) -> "Adjacency":
        """Build a symmetric, de-duplicated, self-loop-free CSR adjacency."""
        import scipy.sparse as sp

        edge_index = np.asarray(edge_index)
        if edge_index.size == 0:
            return cls(np.zeros(num_nodes + 1, dtype=np.int64),
                       np.zeros(0, dtype=np.int32), num_nodes)
        src = np.asarray(edge_index[0], dtype=np.int64)
        dst = np.asarray(edge_index[1], dtype=np.int64)
        keep = src != dst
        src, dst = src[keep], dst[keep]
        # bool data keeps the value array at 1 byte/nnz instead of float64's 8
        u = np.concatenate([src, dst])
        v = np.concatenate([dst, src])
        del src, dst
        mat = sp.coo_matrix(
            (np.ones(u.size, dtype=bool), (u, v)), shape=(num_nodes, num_nodes)
        ).tocsr()
        del u, v
        mat.sum_duplicates()
        mat.sort_indices()
        return cls(mat.indptr.astype(np.int64), mat.indices, num_nodes)

    def neighbors(self, node: int) -> np.ndarray:
        return self.indices[self.indptr[node]:self.indptr[node + 1]]

    @property
    def degree(self) -> np.ndarray:
        return np.diff(self.indptr)

    def has_edge(self, u: int, v: int) -> bool:
        row = self.neighbors(u)
        pos = np.searchsorted(row, v)
        return bool(pos < row.size and row[pos] == v)

    def contains_pairs(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Vectorised membership test over a (small) batch of pairs."""
        return np.array([self.has_edge(int(a), int(b)) for a, b in zip(u, v)], dtype=bool)

    @property
    def num_undirected_edges(self) -> int:
        return int(self.indices.size // 2)


def sample_undirected_pairs(
    edge_index: np.ndarray, k: Optional[int], rng: np.random.Generator
) -> List[Tuple[int, int]]:
    """Sample up to ``k`` distinct undirected pairs straight from an edge array.

    Sampling the raw array avoids materialising all 16M holdout pairs. With a few
    thousand positives the AUC standard error is well under 0.01, so the cap costs
    precision we do not need rather than validity.
    """
    edge_index = np.asarray(edge_index)
    n_edges = edge_index.shape[1]
    if k is None or k >= n_edges:
        cols = np.arange(n_edges)
    else:
        # oversample to absorb self-loops and (u,v)/(v,u) duplicates
        cols = rng.choice(n_edges, size=min(n_edges, int(k * 2.5) + 64), replace=False)
    a = np.asarray(edge_index[0][cols], dtype=np.int64)
    b = np.asarray(edge_index[1][cols], dtype=np.int64)
    keep = a != b
    a, b = a[keep], b[keep]
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    seen = set()
    out: List[Tuple[int, int]] = []
    for x, y in zip(lo, hi):
        key = (int(x), int(y))
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
        if k is not None and len(out) >= k:
            break
    return out


# --------------------------------------------------------------------------- #
# Pair construction
# --------------------------------------------------------------------------- #


@dataclass
class PairSet:
    """A scored pair set. ``label`` is 1 for held-out edges, 0 for negatives."""

    u: np.ndarray
    v: np.ndarray
    label: np.ndarray
    negative_kind: str

    def __len__(self) -> int:
        return int(self.u.size)

    def nodes(self) -> np.ndarray:
        return np.unique(np.concatenate([self.u, self.v]))


def _degree_bin(deg: np.ndarray) -> np.ndarray:
    """Log2 degree bins -- coarse enough to always find matches in a power-law graph."""
    return np.floor(np.log2(np.maximum(deg, 1))).astype(np.int64)


def sample_negatives(
    positives: Sequence[Tuple[int, int]],
    background: Adjacency,
    holdout: Adjacency,
    kind: str,
    rng: np.random.Generator,
    max_tries: int = 200,
) -> List[Tuple[int, int]]:
    """For each positive ``(u, v)`` draw one negative ``(u, v')``.

    ``v'`` is never adjacent to ``u`` in the background *or* the holdout graph, so
    a "negative" is never a true edge that merely sits in the other split.

    kind:
      ``random``         -- v' uniform over all nodes.
      ``degree_matched`` -- v' drawn from the same log2-degree bin as v. This is
                            the headline setting: it removes the degree confound
                            that made the old evaluator scoreable without any
                            pairwise reasoning.
      ``hard_2hop``      -- v' two hops from u in the background graph but not a
                            direct neighbour.
    """
    if kind not in {"random", "degree_matched", "hard_2hop"}:
        raise ValueError(f"unknown negative kind {kind!r}")

    deg = background.degree
    bins = _degree_bin(deg)
    n_nodes = background.num_nodes
    # Degree-bin pools are built lazily and cached: a full per-bin index over 23M
    # nodes costs ~184MB and most of it is never consulted.
    bin_cache: Dict[int, np.ndarray] = {}

    def pool_for_bin(b: int) -> np.ndarray:
        if b not in bin_cache:
            bin_cache[b] = np.flatnonzero(bins == b)
        return bin_cache[b]

    def is_forbidden(u: int, cand: int) -> bool:
        # Membership by binary search on the sorted CSR row -- materialising a
        # hub's neighbour set (millions of entries) is not an option here.
        return cand == u or background.has_edge(u, cand) or holdout.has_edge(u, cand)

    # Caps bounding the 2-hop expansion; a hub can have millions of neighbours, so
    # the shell is sampled rather than enumerated.
    FANOUT, SHELL = 64, 64

    negatives: List[Tuple[int, int]] = []
    for (u, v) in positives:
        if kind == "degree_matched":
            pool = pool_for_bin(int(bins[v]))
            if pool.size == 0:
                pool = np.arange(n_nodes)
        elif kind == "hard_2hop":
            nbrs = background.neighbors(u)
            if nbrs.size > FANOUT:
                nbrs = nbrs[rng.choice(nbrs.size, size=FANOUT, replace=False)]
            shell: List[int] = []
            for nb in nbrs:
                row = background.neighbors(int(nb))
                if row.size > SHELL:
                    row = row[rng.choice(row.size, size=SHELL, replace=False)]
                shell.extend(int(c) for c in row)
            pool = np.unique(np.asarray(shell, dtype=np.int64)) if shell else np.arange(n_nodes)
        else:
            pool = np.arange(n_nodes)

        chosen = -1
        for _ in range(max_tries):
            cand = int(pool[rng.integers(len(pool))])
            if not is_forbidden(int(u), cand):
                chosen = cand
                break
        if chosen < 0:
            # Matched pool exhausted; fall back to a uniform draw so the pair set
            # stays balanced rather than silently dropping the positive.
            for _ in range(max_tries):
                cand = int(rng.integers(n_nodes))
                if not is_forbidden(int(u), cand):
                    chosen = cand
                    break
        if chosen < 0:
            continue
        negatives.append((int(u), chosen))
    return negatives


def build_pair_set(
    background: Adjacency,
    holdout: Adjacency,
    negative_kind: str,
    rng: np.random.Generator,
    max_positives: Optional[int] = None,
    holdout_edge_index: Optional[np.ndarray] = None,
) -> PairSet:
    """Positives = held-out edges; one matched negative per positive.

    ``holdout_edge_index`` samples positives directly from the raw edge array,
    which is the only workable route on the full graphs.
    """
    if holdout_edge_index is not None:
        pos_pairs = sample_undirected_pairs(holdout_edge_index, max_positives, rng)
    else:
        # small-graph path (tests): reconstruct pairs from the CSR upper triangle
        rows = np.repeat(np.arange(holdout.num_nodes), np.diff(holdout.indptr))
        cols = holdout.indices.astype(np.int64)
        upper = rows < cols
        pos_pairs = list(zip(rows[upper].tolist(), cols[upper].tolist()))
        if max_positives is not None and len(pos_pairs) > max_positives:
            idx = rng.choice(len(pos_pairs), size=max_positives, replace=False)
            pos_pairs = [pos_pairs[i] for i in np.sort(idx)]
    if not pos_pairs:
        raise ValueError("holdout graph has no edges -- nothing to evaluate")

    neg_pairs = sample_negatives(pos_pairs, background, holdout, negative_kind, rng)

    u = np.array([p[0] for p in pos_pairs] + [p[0] for p in neg_pairs], dtype=np.int64)
    v = np.array([p[1] for p in pos_pairs] + [p[1] for p in neg_pairs], dtype=np.int64)
    label = np.concatenate([np.ones(len(pos_pairs), dtype=np.int64),
                            np.zeros(len(neg_pairs), dtype=np.int64)])
    return PairSet(u=u, v=v, label=label, negative_kind=negative_kind)


# --------------------------------------------------------------------------- #
# Topology heuristics -- the floors the encoder has to clear to mean anything
# --------------------------------------------------------------------------- #


def heuristic_scores(name: str, pairs: PairSet, adj: Adjacency) -> np.ndarray:
    """Classical link-prediction heuristics on the background graph."""
    deg = adj.degree
    out = np.zeros(len(pairs), dtype=np.float64)
    for i, (u, v) in enumerate(zip(pairs.u, pairs.v)):
        nu = adj.neighbors(int(u))
        nv = adj.neighbors(int(v))
        if name == "preferential_attachment":
            out[i] = float(deg[u]) * float(deg[v])
            continue
        common = np.intersect1d(nu, nv, assume_unique=True)
        if name == "common_neighbors":
            out[i] = common.size
        elif name == "adamic_adar":
            if common.size:
                d = deg[common].astype(np.float64)
                # degree-1 commons contribute 1/log(1)=inf; guard as in standard impls
                out[i] = float(np.sum(1.0 / np.log(np.maximum(d, 2.0))))
        elif name == "jaccard":
            union = np.union1d(nu, nv).size
            out[i] = common.size / union if union else 0.0
        else:
            raise ValueError(f"unknown heuristic {name!r}")
    return out


HEURISTICS = ("common_neighbors", "adamic_adar", "preferential_attachment", "jaccard")


# --------------------------------------------------------------------------- #
# Metrics, with orientation locked on validation
# --------------------------------------------------------------------------- #


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based ROC-AUC with correct tie handling (no sklearn dependency)."""
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0  # average rank for ties
        i = j + 1
    sum_pos = float(ranks[labels == 1].sum())
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="mergesort")
    lab = np.asarray(labels)[order]
    n_pos = int((lab == 1).sum())
    if n_pos == 0:
        return float("nan")
    cum_tp = np.cumsum(lab == 1)
    precision = cum_tp / np.arange(1, len(lab) + 1)
    return float(precision[lab == 1].sum() / n_pos)


def hits_at_k(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="mergesort")
    lab = np.asarray(labels)[order][:k]
    n_pos = int((np.asarray(labels) == 1).sum())
    return float((lab == 1).sum() / min(k, n_pos)) if n_pos else float("nan")


@dataclass
class ScoreReport:
    name: str
    auc: float
    average_precision: float
    hits_at_50: float
    orientation: int
    n_pairs: int
    n_positive: int

    def as_dict(self) -> dict:
        return asdict(self)


def evaluate_scores(
    name: str,
    labels: np.ndarray,
    scores: np.ndarray,
    val_mask: np.ndarray,
) -> ScoreReport:
    """Lock the sign on validation, then report locked metrics on test.

    The old evaluator reported raw AUCs, so an encoder whose signal ran the wrong
    way surfaced as a sub-chance number (e.g. CLFP 0.17) that was then read as
    "worse than chance" rather than "inverted". Choosing the orientation on a
    disjoint validation split makes the test number honest in both directions.
    """
    val_auc = roc_auc(labels[val_mask], scores[val_mask])
    orientation = -1 if (not math.isnan(val_auc) and val_auc < 0.5) else 1
    test_mask = ~val_mask
    oriented = scores * orientation
    return ScoreReport(
        name=name,
        auc=roc_auc(labels[test_mask], oriented[test_mask]),
        average_precision=average_precision(labels[test_mask], oriented[test_mask]),
        hits_at_50=hits_at_k(labels[test_mask], oriented[test_mask], 50),
        orientation=orientation,
        n_pairs=int(test_mask.sum()),
        n_positive=int((labels[test_mask] == 1).sum()),
    )


def split_val_mask(pairs: PairSet, rng: np.random.Generator, val_frac: float = 0.3) -> np.ndarray:
    """Random pair-level split, stratified by label."""
    mask = np.zeros(len(pairs), dtype=bool)
    for lab in (0, 1):
        idx = np.flatnonzero(pairs.label == lab)
        n_val = int(round(val_frac * idx.size))
        if n_val:
            mask[rng.choice(idx, size=n_val, replace=False)] = True
    return mask


def balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    """Balanced accuracy for binary arrays, without an sklearn dependency."""
    labels = np.asarray(labels, dtype=np.int8)
    predictions = np.asarray(predictions, dtype=np.int8)
    recalls = []
    for label in (0, 1):
        mask = labels == label
        if mask.any():
            recalls.append(float(np.mean(predictions[mask] == label)))
    return float(np.mean(recalls)) if recalls else float("nan")


def lock_decision_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    val_mask: np.ndarray,
    orientation: int,
) -> tuple[float, float]:
    """Choose a binary decision threshold on validation only.

    Pair-LP's cosine score has no intrinsic probability threshold.  For a
    correct/incorrect audit we orient it using validation AUC, then select the
    threshold that maximises validation balanced accuracy.  The locked threshold
    is applied unchanged to test pairs.  Returning the middle tied optimum keeps
    the choice deterministic without favouring systematically high or low recall.
    """
    labels = np.asarray(labels, dtype=np.int8)
    oriented = np.asarray(scores, dtype=np.float64) * int(orientation)
    mask = np.asarray(val_mask, dtype=bool)
    y = labels[mask]
    s = oriented[mask]
    if y.size == 0 or np.unique(y).size < 2:
        raise ValueError("threshold locking requires validation examples from both classes")

    unique = np.unique(s)
    if unique.size == 1:
        candidates = unique
    else:
        mids = (unique[:-1] + unique[1:]) / 2.0
        eps = max(1.0, abs(float(unique[0])), abs(float(unique[-1]))) * 1e-12
        candidates = np.concatenate(([unique[0] - eps], mids, [unique[-1] + eps]))

    values = np.asarray([
        balanced_accuracy(y, (s >= threshold).astype(np.int8))
        for threshold in candidates
    ])
    best = float(np.nanmax(values))
    tied = candidates[np.isclose(values, best, rtol=0.0, atol=1e-12)]
    return float(np.median(tied)), best


# --------------------------------------------------------------------------- #
# Pair scoring from endpoint embeddings
# --------------------------------------------------------------------------- #


def pair_scores(emb: np.ndarray, pairs: PairSet, kind: str = "cosine") -> np.ndarray:
    """Symmetric score over the two endpoint embeddings.

    ``emb`` is indexed by node id. Both endpoints enter the score -- this is the
    property the old evaluator lacked.
    """
    hu = emb[pairs.u]
    hv = emb[pairs.v]
    if kind == "dot":
        return np.einsum("ij,ij->i", hu, hv)
    if kind == "cosine":
        nu = np.linalg.norm(hu, axis=1)
        nv = np.linalg.norm(hv, axis=1)
        denom = np.maximum(nu * nv, 1e-12)
        return np.einsum("ij,ij->i", hu, hv) / denom
    raise ValueError(f"unknown score kind {kind!r}")


# --------------------------------------------------------------------------- #
# Sanity gates
# --------------------------------------------------------------------------- #


def endpoint_permutation_auc(
    emb: np.ndarray, pairs: PairSet, rng: np.random.Generator, kind: str = "cosine"
) -> float:
    """Shuffle v across pairs. A pair-conditioned scorer must collapse to ~0.5.

    A scorer that ignores one endpoint is unaffected by this shuffle, so this is
    the direct regression test for defect (1).
    """
    permuted = PairSet(u=pairs.u, v=rng.permutation(pairs.v),
                       label=pairs.label, negative_kind=pairs.negative_kind)
    return roc_auc(permuted.label, pair_scores(emb, permuted, kind))


def endpoint_sensitivity(emb: np.ndarray, pairs: PairSet, kind: str = "cosine") -> float:
    """Fraction of pairs whose score changes when one endpoint is replaced.

    The discriminating value is **zero vs nonzero**: a scorer that ignores an
    endpoint (the old center-blind path) gives exactly 0.0. It approaches 1.0 only
    for continuous embeddings; a low-cardinality embedding (e.g. one-hot over a
    handful of communities) yields many coincidental ties and scores well below 1
    while still being genuinely pair-conditioned. Read it as a gate on 0, not as a
    quality score.
    """
    if len(pairs) < 2:
        return float("nan")
    swapped_v = np.roll(pairs.v, 1)
    # Only count pairs where the substitution actually changed the endpoint;
    # repeated endpoints across pairs otherwise register as spurious insensitivity.
    changed = swapped_v != pairs.v
    if not changed.any():
        return float("nan")
    rolled = PairSet(u=pairs.u, v=swapped_v,
                     label=pairs.label, negative_kind=pairs.negative_kind)
    base = pair_scores(emb, pairs, kind)
    alt = pair_scores(emb, rolled, kind)
    differs = ~np.isclose(base, alt, rtol=1e-9, atol=1e-12)
    return float(np.mean(differs[changed]))


def leakage_check(background: Adjacency, pairs: PairSet) -> int:
    """Count scored positive edges that are still present in the background graph.

    Checks the pairs actually being scored rather than all 16M holdout edges --
    same guarantee where it matters (the encoder must not have aggregated over an
    edge it is asked to predict), at a cost that does not scale with graph size.
    """
    pos = pairs.label == 1
    if not pos.any():
        return 0
    return int(background.contains_pairs(pairs.u[pos], pairs.v[pos]).sum())


# --------------------------------------------------------------------------- #
# Frozen-encoder node embeddings
# --------------------------------------------------------------------------- #


def embed_nodes(
    model,
    subgraph_dataset,
    node_ids: Sequence[int],
    device: str = "cuda",
    batch_size: int = 256,
    return_context: bool = False,
    context_size: int = 3,
):
    """Pooled subgraph embedding per node, using the model's own encoder stack.

    Mirrors ``SingleLayerGeneralGNN.forward`` up to ``final_input_mlp`` but skips
    the metagraph layer. That skip is faithful, not a shortcut: the eval path runs
    with ``zero_shot=True``, under which ``forward_metagraph`` already bypasses
    message passing and returns its inputs unchanged.

    Returns an array indexed by position in ``node_ids``.  With
    ``return_context=True``, also returns ``{center: [sampled neighbour ids]}``
    captured from the exact subgraph instance used for that embedding.
    """
    import torch
    from torch_geometric.data import Batch
    from models.layer_classes import (
        BackgroundGNNLayer,
        MetagraphLayer,
        SupernodeAggrLayer,
        SupernodeToBgGraphLayer,
    )

    model.eval()
    model.to(device)
    skip_path = bool(model.params.get("skip_path", False))
    out: List[np.ndarray] = []
    contexts: Dict[int, List[int]] = {}

    with torch.no_grad():
        for start in range(0, len(node_ids), batch_size):
            chunk = [int(n) for n in node_ids[start:start + batch_size]]
            graphs = [subgraph_dataset[n] for n in chunk]
            if return_context:
                for center, sampled in zip(chunk, graphs):
                    global_ids = getattr(sampled, "global_node_ids", None)
                    if global_ids is None:
                        contexts[center] = []
                        continue
                    contexts[center] = [
                        int(node)
                        for node in global_ids.detach().cpu().reshape(-1).tolist()
                        if int(node) >= 0 and int(node) != center
                    ][:max(0, int(context_size))]
            graph = Batch.from_data_list(graphs).to(device)

            supernode_idx = graph.supernode + graph.ptr[:-1]
            graph.x = model.initial_input_mlp(graph.x)
            if model.txt_dropout is not None:
                graph.x = model.txt_dropout(graph.x)
            x_orig = graph.x.clone()

            x_input = None
            for module in model.layer_list:
                if isinstance(module, MetagraphLayer):
                    continue  # zero-shot: identity on x_input (general_gnn.py:70)
                if isinstance(module, SupernodeAggrLayer):
                    x_input = module.forward(graph.x, graph.edge_index_supernode,
                                             supernode_idx, graph.batch)
                    graph.x = graph.x.clone()
                    graph.x[supernode_idx] = x_input
                elif isinstance(module, BackgroundGNNLayer):
                    new_x = module.forward(
                        x_orig, graph.x, graph.edge_index.long(),
                        graph.edge_attr if "edge_attr" in graph else None,
                        graph.edge_index_supernode, graph.ptr[:-1], graph.batch)
                    graph.x = (graph.x + new_x
                               if skip_path and new_x.shape == graph.x.shape else new_x)
                elif isinstance(module, SupernodeToBgGraphLayer):
                    if x_input is None:
                        raise RuntimeError(
                            "SupernodeToBgGraphLayer reached before any supernode "
                            "aggregation produced x_input")
                    new_x = module.forward(graph.x, x_input,
                                           graph.edge_index_supernode, supernode_idx,
                                           graph.batch)
                    graph.x = graph.x + new_x if skip_path else new_x
                else:
                    raise ValueError(f"unknown layer type {type(module)}")

            if x_input is None:
                raise RuntimeError(
                    "encoder produced no pooled embedding -- layer string must "
                    "contain a supernode aggregation layer (e.g. 'S,U,M')")
            out.append(model.final_input_mlp(x_input).float().cpu().numpy())

    packed = np.concatenate(out, axis=0)
    return (packed, contexts) if return_context else packed


class NodeEmbeddings:
    """Sparse node->embedding map that indexes like a dense array.

    A dense ``num_nodes x d`` table is impossible at real scale (23M x 256 floats
    = 23.5GB) when only a few thousand nodes are ever scored. This stores just the
    embedded rows plus an int32 lookup, while supporting ``emb[node_array]`` so
    call sites read the same as with an ndarray.
    """

    __slots__ = ("table", "index", "num_nodes")

    def __init__(self, table: np.ndarray, nodes: np.ndarray, num_nodes: int):
        self.table = table
        self.num_nodes = num_nodes
        self.index = np.full(num_nodes, -1, dtype=np.int32)
        self.index[nodes] = np.arange(nodes.size, dtype=np.int32)

    def __getitem__(self, nodes) -> np.ndarray:
        rows = self.index[nodes]
        if np.any(rows < 0):
            missing = np.asarray(nodes)[rows < 0][:5]
            raise KeyError(f"no embedding for node(s) {missing.tolist()}")
        return self.table[rows]

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.num_nodes, self.table.shape[1])


def embeddings_by_node(
    model, subgraph_dataset, nodes: np.ndarray, num_nodes: int, **kwargs
):
    """Embed ``nodes`` and wrap them in a node-indexed sparse map."""
    packed = embed_nodes(model, subgraph_dataset, nodes.tolist(), **kwargs)
    if isinstance(packed, tuple):
        table, contexts = packed
        return NodeEmbeddings(table, np.asarray(nodes), num_nodes), contexts
    return NodeEmbeddings(packed, np.asarray(nodes), num_nodes)


# --------------------------------------------------------------------------- #
# Full evaluation over one graph
# --------------------------------------------------------------------------- #


def evaluate_graph(
    background: Adjacency,
    holdout: Adjacency,
    embeddings: Optional[np.ndarray],
    raw_features: Optional[np.ndarray],
    negative_kind: str,
    seed: int,
    max_positives: Optional[int] = None,
    score_kind: str = "cosine",
    holdout_edge_index: Optional[np.ndarray] = None,
) -> dict:
    """Score one graph under one negative-sampling regime.

    Returns encoder reports alongside the heuristic and raw-feature floors, plus
    the sanity gates. A caller should treat any result whose gates fail as void.
    """
    rng = np.random.default_rng(seed)
    pairs = build_pair_set(background, holdout, negative_kind, rng,
                           max_positives=max_positives,
                           holdout_edge_index=holdout_edge_index)
    val_mask = split_val_mask(pairs, rng)

    reports: List[ScoreReport] = []
    gates: Dict[str, float] = {}

    if embeddings is not None:
        scores = pair_scores(embeddings, pairs, score_kind)
        reports.append(evaluate_scores(f"encoder_{score_kind}", pairs.label, scores, val_mask))
        gates["endpoint_permutation_auc"] = endpoint_permutation_auc(
            embeddings, pairs, np.random.default_rng(seed + 1), score_kind)
        gates["endpoint_sensitivity"] = endpoint_sensitivity(embeddings, pairs, score_kind)

    if raw_features is not None:
        reports.append(evaluate_scores(
            "raw_feature_cosine", pairs.label,
            pair_scores(raw_features, pairs, "cosine"), val_mask))

    for h in HEURISTICS:
        reports.append(evaluate_scores(
            h, pairs.label, heuristic_scores(h, pairs, background), val_mask))

    gates["holdout_leakage_edges"] = float(leakage_check(background, pairs))

    return {
        "negative_kind": negative_kind,
        "score_kind": score_kind,
        "n_pairs": len(pairs),
        "n_positive": int((pairs.label == 1).sum()),
        "gates": gates,
        "reports": [r.as_dict() for r in reports],
    }


# --------------------------------------------------------------------------- #
# Synthetic self-test (Gate 0, item 7)
# --------------------------------------------------------------------------- #


def _synthetic_graph(n_per_block: int = 60, n_blocks: int = 4, p_in: float = 0.25,
                     p_out: float = 0.01, seed: int = 0):
    """Stochastic block model with a known edge mechanism.

    Edges are driven by block membership, so an embedding that encodes the block
    must beat chance and common-neighbour must beat preferential-attachment.
    Anything that fails this ordering is a broken evaluator, not a weak encoder.
    """
    rng = np.random.default_rng(seed)
    n = n_per_block * n_blocks
    block = np.repeat(np.arange(n_blocks), n_per_block)
    src, dst = [], []
    for i in range(n):
        for j in range(i + 1, n):
            p = p_in if block[i] == block[j] else p_out
            if rng.random() < p:
                src.append(i)
                dst.append(j)
    edges = np.array([src, dst], dtype=np.int64)
    # 85/15 background/holdout split over undirected pairs
    perm = rng.permutation(edges.shape[1])
    n_hold = max(20, int(0.15 * edges.shape[1]))
    hold_idx, bg_idx = perm[:n_hold], perm[n_hold:]
    background = Adjacency.from_edge_index(edges[:, bg_idx], n)
    holdout = Adjacency.from_edge_index(edges[:, hold_idx], n)
    return background, holdout, block, n


def self_test(verbose: bool = True) -> bool:
    """Validate the evaluator against a graph whose mechanism we control."""
    ok = True

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        ok &= bool(cond)
        if verbose:
            print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")

    background, holdout, block, n = _synthetic_graph()
    rng = np.random.default_rng(0)

    if verbose:
        print("Gate 0 self-test -- synthetic stochastic block model")
        print(f"  nodes={n} background_edges={background.indices.size // 2} "
              f"holdout_edges={holdout.indices.size // 2}")

    _probe = build_pair_set(background, holdout, "random",
                            np.random.default_rng(99), max_positives=200)
    check("no holdout leakage into background", leakage_check(background, _probe) == 0)

    # An oracle embedding: one-hot block identity. Same-block pairs score 1.
    oracle = np.eye(4, dtype=np.float32)[block]
    # An endpoint-blind embedding: every node identical -> no pairwise information.
    blind = np.ones((n, 4), dtype=np.float32)

    # Expected AUC floor for the block oracle. Under hard_2hop the negatives are
    # drawn from u's own 2-hop shell, which in a block model lies overwhelmingly
    # inside u's block -- so block identity (and common-neighbour, since a 2-hop
    # negative shares a neighbour with u by construction) barely separates
    # positives from negatives. Hard negatives are a punishing secondary
    # condition, not a headline; degree_matched is the primary read.
    oracle_floor = {"random": 0.75, "degree_matched": 0.75, "hard_2hop": 0.55}
    for kind in ("random", "degree_matched", "hard_2hop"):
        res = evaluate_graph(background, holdout, oracle, None, kind, seed=1,
                             max_positives=400)
        by_name = {r["name"]: r for r in res["reports"]}
        auc = by_name["encoder_cosine"]["auc"]
        check(f"oracle beats chance ({kind})", auc > oracle_floor[kind], f"AUC={auc:.3f}")
        # Discrete one-hot oracle -> many coincidental ties; the invariant is
        # "depends on the endpoint at all", i.e. clearly nonzero.
        check(f"oracle is endpoint-sensitive ({kind})",
              res["gates"]["endpoint_sensitivity"] > 0.1,
              f"{res['gates']['endpoint_sensitivity']:.3f}")
        perm_auc = res["gates"]["endpoint_permutation_auc"]
        check(f"endpoint permutation destroys signal ({kind})",
              abs(perm_auc - 0.5) < 0.12, f"AUC={perm_auc:.3f}")
        cn = by_name["common_neighbors"]["auc"]
        if kind == "hard_2hop":
            check("hard negatives neutralise common-neighbour", cn < 0.65,
                  f"AUC={cn:.3f}")
        else:
            check(f"common-neighbour beats chance ({kind})", cn > 0.6, f"AUC={cn:.3f}")

    # Continuous embedding: sensitivity must be essentially 1.0 when ties are
    # measure-zero. This is the strict form of the endpoint-dependence gate.
    cont = np.random.default_rng(7).normal(size=(n, 16)).astype(np.float32)
    res_cont = evaluate_graph(background, holdout, cont, None, "degree_matched",
                              seed=5, max_positives=300)
    check("continuous embedding is fully endpoint-sensitive",
          res_cont["gates"]["endpoint_sensitivity"] > 0.99,
          f"{res_cont['gates']['endpoint_sensitivity']:.3f}")
    rand_auc = {r["name"]: r for r in res_cont["reports"]}["encoder_cosine"]["auc"]
    check("random embedding scores near chance", abs(rand_auc - 0.5) < 0.12,
          f"AUC={rand_auc:.3f}")

    # Degree matching must neutralise a degree-only scorer, while random negatives
    # leave it exploitable -- this is the regression test for defect (3).
    deg_emb = background.degree.astype(np.float32).reshape(-1, 1)
    deg_emb = np.hstack([deg_emb, np.ones_like(deg_emb)])
    auc_random = None
    auc_matched = None
    for kind in ("random", "degree_matched"):
        res = evaluate_graph(background, holdout, deg_emb, None, kind, seed=2,
                             max_positives=400)
        auc = {r["name"]: r for r in res["reports"]}["encoder_cosine"]["auc"]
        if kind == "random":
            auc_random = auc
        else:
            auc_matched = auc
    check("degree-matched negatives neutralise a degree-only scorer",
          abs(auc_matched - 0.5) < abs(auc_random - 0.5) + 1e-9,
          f"random={auc_random:.3f} matched={auc_matched:.3f}")

    # A constant (endpoint-blind) embedding must be exactly chance.
    res = evaluate_graph(background, holdout, blind, None, "degree_matched", seed=3,
                         max_positives=300)
    blind_auc = {r["name"]: r for r in res["reports"]}["encoder_cosine"]["auc"]
    check("endpoint-blind embedding scores chance", abs(blind_auc - 0.5) < 1e-6,
          f"AUC={blind_auc:.3f}")
    check("endpoint-blind embedding fails sensitivity gate",
          res["gates"]["endpoint_sensitivity"] < 1e-9,
          f"{res['gates']['endpoint_sensitivity']:.3f}")

    # Metric implementations against known values.
    check("AUC of a perfect ranking is 1.0",
          abs(roc_auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) - 1.0) < 1e-12)
    check("AUC of an inverted ranking is 0.0",
          abs(roc_auc(np.array([0, 0, 1, 1]), np.array([0.9, 0.8, 0.2, 0.1]))) < 1e-12)
    check("AUC of all-tied scores is 0.5",
          abs(roc_auc(np.array([0, 1, 0, 1]), np.array([0.5, 0.5, 0.5, 0.5])) - 0.5) < 1e-12)

    # Orientation locking must rescue a consistently inverted scorer.
    labels = np.array([1] * 50 + [0] * 50)
    inverted = -labels.astype(np.float64) + np.random.default_rng(4).normal(0, 0.05, 100)
    vmask = np.zeros(100, dtype=bool)
    vmask[:15] = True
    vmask[50:65] = True
    rep = evaluate_scores("inverted", labels, inverted, vmask)
    check("orientation lock recovers an inverted scorer",
          rep.orientation == -1 and rep.auc > 0.95,
          f"orientation={rep.orientation} AUC={rep.auc:.3f}")

    if verbose:
        print(f"\n{'ALL CHECKS PASSED' if ok else 'FAILURES PRESENT'}")
    return ok


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _load_graph_views(graph_path: str, background_view: str, holdout_view: str):
    """Load a prodigy graph artifact and extract the two named edge views."""
    import torch
    raw = torch.load(graph_path, map_location="cpu", weights_only=False)
    graph = raw["graph"] if isinstance(raw, dict) and "graph" in raw else raw

    def get_view(name: str):
        views = getattr(graph, "edge_index_views", None)
        if isinstance(views, dict) and name in views:
            return views[name]
        legacy = f"edge_index_{name}"
        if hasattr(graph, legacy):
            return getattr(graph, legacy)
        raise KeyError(
            f"edge view {name!r} not found; available: "
            f"{sorted(views) if isinstance(views, dict) else 'none'}. "
            "Run scripts/graph_construction/enrich_all_graphs.sh to add "
            "static_background/static_holdout.")

    n = int(graph.num_nodes)
    return (Adjacency.from_edge_index(np.asarray(get_view(background_view)), n),
            Adjacency.from_edge_index(np.asarray(get_view(holdout_view)), n),
            graph)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true",
                    help="run the synthetic-graph gate and exit (no checkpoint needed)")
    ap.add_argument("--graph", help="path to a prodigy graph .pt artifact")
    ap.add_argument("--checkpoint", help="frozen encoder checkpoint; omit for heuristics only")
    ap.add_argument("--background-view", default="static_background")
    ap.add_argument("--holdout-view", default="static_holdout")
    ap.add_argument("--negative-kinds", default="degree_matched,random,hard_2hop")
    ap.add_argument("--score-kind", default="cosine", choices=("cosine", "dot"))
    ap.add_argument("--max-positives", type=int, default=2000,
                    help="cap on held-out edges scored (keeps embedding cost bounded)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", help="write results JSON here")
    args = ap.parse_args(argv)

    if args.self_test:
        return 0 if self_test() else 1

    if not args.graph:
        ap.error("--graph is required unless --self-test is given")

    background, holdout, graph = _load_graph_views(
        args.graph, args.background_view, args.holdout_view)

    raw_features = None
    if hasattr(graph, "x") and graph.x is not None:
        raw_features = np.asarray(graph.x, dtype=np.float32)

    embeddings = None
    if args.checkpoint:
        raise NotImplementedError(
            "checkpoint loading is wired in load_frozen_encoder(); see "
            "scripts/eval/pair_link_eval_ckpt.py for the Tucker entry point")

    results = []
    for kind in args.negative_kinds.split(","):
        results.append(evaluate_graph(
            background, holdout, embeddings, raw_features, kind.strip(),
            seed=args.seed, max_positives=args.max_positives,
            score_kind=args.score_kind))

    payload = {"graph": args.graph, "checkpoint": args.checkpoint, "results": results}
    text = json.dumps(payload, indent=2)
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
