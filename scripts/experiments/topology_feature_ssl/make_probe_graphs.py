#!/usr/bin/env python3
"""Generate planted single-rule synthetic graphs for the capability probes (T3).

Each graph isolates ONE structural primitive as a per-node binary label; the node
FEATURES never leak the label, so a linear probe on the frozen center-node
representation succeeds only if the encoder actually represents that primitive.
The graphs are written in the covid19_twitter dict-format so they flow through the
existing frozen-encoder eval path (classification linear-probe = the probe score).

Rules (README "Capability probes"; the encoder is 1-hop and the sampler
symmetrizes direction, so every rule is 1-hop over the undirected neighborhood
except the two directional ones, which is the point):

  count_threshold : y = 1[ total_degree(v) >= median ]        (random feats)
  in_degree       : y = 1[ in_degree(v)   >= median ]         (random feats; needs
                        direction, which survives only in edge_attr sign)
  out_degree      : y = 1[ out_degree(v)  >= median ]         (random feats)
  existence       : y = 1[ exists neighbor with marker A ]    (marker feats)
  conjunction     : y = 1[ (exists nbr A) AND (exists nbr B) ](marker feats; binding)

Expected reading: mean-agg B0 fails count/degree (mean is count-blind) and only
weakly detects existence; E1 (structural inputs) trivially passes the degree
rules by passthrough; E2 (sum/PNA + directed) passes count/existence and recovers
in/out-degree; conjunction needs cross-neighbor binding (hardest).

Run (anywhere with torch):
    python scripts/experiments/topology_feature_ssl/make_probe_graphs.py \
        --out-dir /dataMeR1/phil/data/synthetic_probes/graphs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

FEATURE_DIM = 768
RULES = ["count_threshold", "in_degree", "out_degree", "existence", "conjunction"]
# Per-rule marker rate, tuned for ~balanced labels at mean_out_deg=5 (undirected
# avg degree ~10 after the sampler symmetrizes): existence wants P(>=1 A-nbr)~0.5
# so (1-p)^10~0.5 -> p~0.065; conjunction wants P(>=1 A)P(>=1 B)~0.5 -> each ~0.71
# -> p~0.13. A single --marker-p overrides both.
RULE_MARKER_P = {"existence": 0.065, "conjunction": 0.13}


def _random_digraph(n: int, mean_out_deg: float, gen: torch.Generator) -> torch.Tensor:
    """Directed edges with heterogeneous degree: out-degree ~ Poisson(mean),
    targets uniform (no self-loops). Returns edge_index [2, E]."""
    out_deg = torch.poisson(torch.full((n,), float(mean_out_deg)), generator=gen).long()
    srcs, dsts = [], []
    for v in range(n):
        d = int(out_deg[v].item())
        if d == 0:
            continue
        tgt = torch.randint(0, n, (d,), generator=gen)
        tgt = tgt[tgt != v]
        srcs.append(torch.full((tgt.numel(),), v, dtype=torch.long))
        dsts.append(tgt)
    if not srcs:
        return torch.empty(2, 0, dtype=torch.long)
    edge_index = torch.stack([torch.cat(srcs), torch.cat(dsts)], dim=0)
    return torch.unique(edge_index, dim=1)


def _degrees(edge_index: torch.Tensor, n: int):
    in_deg = torch.zeros(n); out_deg = torch.zeros(n)
    if edge_index.numel():
        out_deg.scatter_add_(0, edge_index[0], torch.ones(edge_index.size(1)))
        in_deg.scatter_add_(0, edge_index[1], torch.ones(edge_index.size(1)))
    return in_deg, out_deg


def _undirected_neighbors(edge_index: torch.Tensor, n: int) -> list[set[int]]:
    nbrs: list[set[int]] = [set() for _ in range(n)]
    ei = edge_index.tolist()
    for u, v in zip(ei[0], ei[1]):
        nbrs[u].add(v); nbrs[v].add(u)  # symmetrized, matching the sampler
    return nbrs


def _random_features(n: int, gen: torch.Generator) -> torch.Tensor:
    return torch.randn(n, FEATURE_DIM, generator=gen) * 0.1


def _label_balance(y: torch.Tensor) -> float:
    return float(y.float().mean().item())


def build_rule(rule: str, n: int, mean_out_deg: float, marker_p: float,
               marker_mag: float, seed: int) -> dict:
    gen = torch.Generator().manual_seed(seed)
    edge_index = _random_digraph(n, mean_out_deg, gen)
    in_deg, out_deg = _degrees(edge_index, n)
    x = _random_features(n, gen)

    if rule == "count_threshold":
        deg = in_deg + out_deg
        y = (deg >= deg.median()).long()
    elif rule == "in_degree":
        y = (in_deg >= in_deg.median()).long()
    elif rule == "out_degree":
        y = (out_deg >= out_deg.median()).long()
    elif rule in ("existence", "conjunction"):
        nbrs = _undirected_neighbors(edge_index, n)
        has_A = torch.rand(n, generator=gen) < marker_p
        x[has_A, 0] += marker_mag
        if rule == "existence":
            y = torch.tensor([int(any(has_A[j] for j in nbrs[v])) for v in range(n)]).long()
        else:
            has_B = torch.rand(n, generator=gen) < marker_p
            # keep A/B disjoint so a single neighbor can't satisfy both
            has_B = has_B & ~has_A
            x[has_B, 1] += marker_mag
            y = torch.tensor([
                int(any(has_A[j] for j in nbrs[v]) and any(has_B[j] for j in nbrs[v]))
                for v in range(n)
            ]).long()
    else:
        raise ValueError(f"unknown rule {rule}")

    return {
        "x": x,
        "edge_index": edge_index,
        "y": y,
        "label_names": [f"not_{rule}", rule],
        "label_type": "classification",
        "feature_names": [f"emb_{i}" for i in range(FEATURE_DIM)],
        "node_targets": {},
        "node_target_names": [],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="/dataMeR1/phil/data/synthetic_probes/graphs")
    ap.add_argument("--n", type=int, default=4000, help="nodes (probe centers) per rule")
    ap.add_argument("--mean-out-deg", type=float, default=5.0)
    ap.add_argument("--marker-p", type=float, default=None,
                    help="override the per-rule marker rate (default: RULE_MARKER_P per rule)")
    ap.add_argument("--marker-mag", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[probes] writing {len(RULES)} rule graphs to {out_dir} (n={args.n})")
    for i, rule in enumerate(RULES):
        marker_p = args.marker_p if args.marker_p is not None else RULE_MARKER_P.get(rule, 0.1)
        raw = build_rule(rule, args.n, args.mean_out_deg, marker_p,
                         args.marker_mag, seed=args.seed + i)
        path = out_dir / f"{rule}.pt"
        torch.save(raw, path)
        bal = _label_balance(raw["y"])
        deg = raw["edge_index"].size(1) / max(1, args.n)
        print(f"[probes] {rule:16s} -> {path.name}  pos_rate={bal:.2f}  "
              f"avg_deg={deg:.1f}  edges={raw['edge_index'].size(1)}")
        if not 0.15 < bal < 0.85:
            print(f"[probes]   WARNING: {rule} label balance {bal:.2f} is skewed; "
                  "consider tuning --marker-p / --mean-out-deg.")
    print("[probes] done. Add these as probe_* datasets in the eval driver and run "
          "run_capability_probes.sh.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
