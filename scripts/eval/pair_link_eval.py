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
    """Undirected CSR adjacency over ``num_nodes`` nodes."""

    indptr: np.ndarray
    indices: np.ndarray
    num_nodes: int

    @classmethod
    def from_edge_index(cls, edge_index: np.ndarray, num_nodes: int) -> "Adjacency":
        """Build a symmetric, de-duplicated, self-loop-free CSR adjacency."""
        if edge_index.size == 0:
            return cls(np.zeros(num_nodes + 1, dtype=np.int64),
                       np.zeros(0, dtype=np.int64), num_nodes)
        src = np.asarray(edge_index[0], dtype=np.int64)
        dst = np.asarray(edge_index[1], dtype=np.int64)
        # symmetrise, drop self-loops, de-duplicate
        u = np.concatenate([src, dst])
        v = np.concatenate([dst, src])
        keep = u != v
        u, v = u[keep], v[keep]
        order = np.lexsort((v, u))
        u, v = u[order], v[order]
        if u.size:
            uniq = np.ones(u.size, dtype=bool)
            uniq[1:] = (u[1:] != u[:-1]) | (v[1:] != v[:-1])
            u, v = u[uniq], v[uniq]
        indptr = np.zeros(num_nodes + 1, dtype=np.int64)
        np.add.at(indptr, u + 1, 1)
        np.cumsum(indptr, out=indptr)
        return cls(indptr, v.astype(np.int64), num_nodes)

    def neighbors(self, node: int) -> np.ndarray:
        return self.indices[self.indptr[node]:self.indptr[node + 1]]

    @property
    def degree(self) -> np.ndarray:
        return np.diff(self.indptr)

    def has_edge(self, u: int, v: int) -> bool:
        return bool(np.isin(v, self.neighbors(u)))

    def edge_set(self) -> set:
        """All undirected pairs as ordered (min, max) tuples."""
        out = set()
        for u in range(self.num_nodes):
            for v in self.neighbors(u):
                a, b = (u, int(v)) if u < v else (int(v), u)
                out.add((a, b))
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
    by_bin: Dict[int, np.ndarray] = {}
    if kind == "degree_matched":
        for b in np.unique(bins):
            by_bin[int(b)] = np.flatnonzero(bins == b)

    negatives: List[Tuple[int, int]] = []
    for (u, v) in positives:
        forbidden = set(background.neighbors(u).tolist())
        forbidden.update(holdout.neighbors(u).tolist())
        forbidden.add(u)

        if kind == "degree_matched":
            pool = by_bin.get(int(bins[v]), np.arange(background.num_nodes))
        elif kind == "hard_2hop":
            two_hop = set()
            for nb in background.neighbors(u):
                two_hop.update(background.neighbors(int(nb)).tolist())
            cand = [c for c in two_hop if c not in forbidden]
            pool = np.asarray(cand, dtype=np.int64) if cand else np.arange(background.num_nodes)
        else:
            pool = np.arange(background.num_nodes)

        chosen = -1
        for _ in range(max_tries):
            cand = int(pool[rng.integers(len(pool))])
            if cand not in forbidden:
                chosen = cand
                break
        if chosen < 0:
            # Exhausted the matched pool; fall back to any valid non-neighbour so
            # the pair set stays balanced. Counted and reported by the caller.
            allowed = np.setdiff1d(np.arange(background.num_nodes),
                                   np.fromiter(forbidden, dtype=np.int64))
            if allowed.size == 0:
                continue
            chosen = int(allowed[rng.integers(allowed.size)])
        negatives.append((int(u), chosen))
    return negatives


def build_pair_set(
    background: Adjacency,
    holdout: Adjacency,
    negative_kind: str,
    rng: np.random.Generator,
    max_positives: Optional[int] = None,
) -> PairSet:
    """Positives = held-out edges; one matched negative per positive."""
    pos_pairs = sorted(holdout.edge_set())
    if not pos_pairs:
        raise ValueError("holdout graph has no edges -- nothing to evaluate")
    if max_positives is not None and len(pos_pairs) > max_positives:
        idx = rng.choice(len(pos_pairs), size=max_positives, replace=False)
        pos_pairs = [pos_pairs[i] for i in np.sort(idx)]

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


def leakage_check(background: Adjacency, holdout: Adjacency) -> int:
    """Count held-out edges still present in the background graph. Must be 0."""
    bg = background.edge_set()
    return sum(1 for e in holdout.edge_set() if e in bg)


# --------------------------------------------------------------------------- #
# Frozen-encoder node embeddings
# --------------------------------------------------------------------------- #


def embed_nodes(
    model,
    subgraph_dataset,
    node_ids: Sequence[int],
    device: str = "cuda",
    batch_size: int = 256,
) -> np.ndarray:
    """Pooled subgraph embedding per node, using the model's own encoder stack.

    Mirrors ``SingleLayerGeneralGNN.forward`` up to ``final_input_mlp`` but skips
    the metagraph layer. That skip is faithful, not a shortcut: the eval path runs
    with ``zero_shot=True``, under which ``forward_metagraph`` already bypasses
    message passing and returns its inputs unchanged.

    Returns an array indexed by position in ``node_ids``.
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

    with torch.no_grad():
        for start in range(0, len(node_ids), batch_size):
            chunk = [int(n) for n in node_ids[start:start + batch_size]]
            graphs = [subgraph_dataset[n] for n in chunk]
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

    return np.concatenate(out, axis=0)


def embeddings_by_node(
    model, subgraph_dataset, nodes: np.ndarray, num_nodes: int, **kwargs
) -> np.ndarray:
    """Dense ``num_nodes x d`` table with rows filled only for ``nodes``."""
    packed = embed_nodes(model, subgraph_dataset, nodes.tolist(), **kwargs)
    table = np.zeros((num_nodes, packed.shape[1]), dtype=np.float32)
    table[nodes] = packed
    return table


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
) -> dict:
    """Score one graph under one negative-sampling regime.

    Returns encoder reports alongside the heuristic and raw-feature floors, plus
    the sanity gates. A caller should treat any result whose gates fail as void.
    """
    rng = np.random.default_rng(seed)
    pairs = build_pair_set(background, holdout, negative_kind, rng,
                           max_positives=max_positives)
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

    gates["holdout_leakage_edges"] = float(leakage_check(background, holdout))

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

    check("no holdout leakage into background", leakage_check(background, holdout) == 0)

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
