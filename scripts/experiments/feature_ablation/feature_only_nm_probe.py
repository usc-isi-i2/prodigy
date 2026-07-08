#!/usr/bin/env python3
"""Feature-only Neighbor-Matching probe: can node features alone do NM?

Motivation. The feature-ablation experiment showed the NM-pretrained model is
permute-invariant — it uses node features only as distinguishers, not content.
Open question: is that because feature *content* is genuinely uninformative for
the NM target (which neighborhood a node belongs to), or because the model
chose to ignore usable signal? If forcing feature use could ever help NM, there
must be neighborhood-discriminative signal in the bios in the first place.

This probe measures exactly that, with NO model and NO training. It rebuilds the
NM episode structure and solves it by prototype nearest-neighbor in raw feature
space:

  * sample `n_way` centers (nodes with enough neighbors);
  * for each center sample `n_shot + n_query` of its (undirected) neighbors,
    labeled by that center;
  * prototype[c] = mean L2-normalized feature of the c support nodes;
  * classify each query to the nearest prototype (cosine); score top-1 acc + AUC.

Controls:
  * `real`     — actual features. If >> chance, bios carry neighborhood signal.
  * `permute`  — features shuffled across all nodes (destroys node<->feature
    binding). Should fall to chance; isolates that any `real` signal is genuine.

Read the result against chance (1/n_way) AND the model's NM accuracy on the same
graph: if feature-only >> chance the content is neighborhood-informative (so
forcing feature use could help NM); if feature-only is far below the model, the
topology channel dominates and the feature headroom for NM is small.

Self-contained (torch + numpy + sklearn); no dependence on the training code.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def load_graph(path: str):
    g = torch.load(path, map_location="cpu")
    if isinstance(g, dict):
        d = g["data"] if ("data" in g and hasattr(g["data"], "x")) else None
        x = d.x if d is not None else g["x"]
        ei = (d.edge_index if d is not None else g["edge_index"])
    else:
        x, ei = g.x, g.edge_index
    return x.float().numpy(), ei.long().numpy()


def build_adjacency(edge_index: np.ndarray, num_nodes: int):
    # undirected neighbor lists ("inout")
    src = np.concatenate([edge_index[0], edge_index[1]])
    dst = np.concatenate([edge_index[1], edge_index[0]])
    order = np.argsort(src, kind="stable")
    src, dst = src[order], dst[order]
    # boundaries per source node
    counts = np.bincount(src, minlength=num_nodes)
    ptr = np.zeros(num_nodes + 1, dtype=np.int64)
    ptr[1:] = np.cumsum(counts)
    return ptr, dst


def neighbors(ptr, dst, node):
    return dst[ptr[node]:ptr[node + 1]]


def l2norm(a, axis=1, eps=1e-8):
    return a / (np.linalg.norm(a, axis=axis, keepdims=True) + eps)


def run(x, ptr, dst, *, n_way, n_shot, n_query, episodes, seed, permute):
    rng = np.random.default_rng(seed)
    num_nodes = x.shape[0]
    feats = x.copy()
    if permute:
        feats = feats[rng.permutation(num_nodes)]
    feats = l2norm(feats)

    deg = np.diff(ptr)
    need = n_shot + n_query
    eligible = np.nonzero(deg >= need)[0]
    if len(eligible) < n_way:
        raise ValueError(f"Only {len(eligible)} eligible centers for n_way={n_way}")

    all_true, all_pred, all_scores = [], [], []
    for _ in range(episodes):
        centers = rng.choice(eligible, size=n_way, replace=False)
        protos = np.zeros((n_way, feats.shape[1]), dtype=np.float32)
        q_feat, q_lab = [], []
        for ci, c in enumerate(centers):
            nb = neighbors(ptr, dst, c)
            pick = rng.choice(nb, size=need, replace=False)
            sup, qry = pick[:n_shot], pick[n_shot:]
            protos[ci] = feats[sup].mean(0)
            q_feat.append(feats[qry])
            q_lab.append(np.full(len(qry), ci))
        protos = l2norm(protos)
        Q = np.concatenate(q_feat)          # (n_way*n_query, d)
        y = np.concatenate(q_lab)
        sims = Q @ protos.T                  # cosine similarities -> class scores
        pred = sims.argmax(1)
        all_true.append(y); all_pred.append(pred); all_scores.append(sims)

    y = np.concatenate(all_true)
    pred = np.concatenate(all_pred)
    scores = np.concatenate(all_scores)
    acc = float((pred == y).mean())
    # macro one-vs-rest AUC over the per-episode class scores
    try:
        yoh = np.eye(n_way)[y]
        auc = float(roc_auc_score(yoh, scores, multi_class="ovr", average="macro"))
    except Exception:
        auc = float("nan")
    return {"accuracy": acc, "roc_auc": auc, "chance": 1.0 / n_way,
            "n_query_total": int(len(y))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--name", default="")
    ap.add_argument("--n-way", type=int, default=30)
    ap.add_argument("--n-shot", type=int, default=3)
    ap.add_argument("--n-query", type=int, default=10)
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    name = args.name or args.graph

    x, ei = load_graph(args.graph)
    ptr, dst = build_adjacency(ei, x.shape[0])

    rows = {}
    for cond in ("real", "permute"):
        r = run(x, ptr, dst, n_way=args.n_way, n_shot=args.n_shot, n_query=args.n_query,
                episodes=args.episodes, seed=args.seed, permute=(cond == "permute"))
        rows[cond] = r
        print(f"{name:<18} {cond:<8} acc={r['accuracy']:.4f}  AUC={r['roc_auc']:.4f}  "
              f"(chance {r['chance']:.4f}, queries {r['n_query_total']})")

    payload = {"name": name, "graph": args.graph, "n_way": args.n_way,
               "n_shot": args.n_shot, "n_query": args.n_query, "episodes": args.episodes,
               "seed": args.seed, "results": rows}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
