#!/usr/bin/env python3
"""Are the dims that SEPARATE two graphs the same dims that carry task signal?

Two subspaces have now been measured:
  * separation dims -- where graph A's bio cloud sits apart from graph B's
    (compute_subspace_divergence.py: 8 of 768 dims recover most of proxy-A)
  * coupling dims   -- where a user's bio predicts who they retweet
    (compute_structural_subspace.py: largely shared across graphs)

If they are disjoint, the cross-graph shift is pure nuisance: per-graph centring
removes it for free and nothing task-relevant is lost. If they coincide, then
aligning the clouds would destroy exactly the signal the task runs on, and any
alignment scheme is actively harmful.

Reports observed overlap against the chance baseline k^2/D, plus the rank
correlation of the two per-dim scores.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_graph_divergence import (  # noqa: E402
    DEFAULT_GRAPHS, as_numpy, load_graph, log, sample_feature_rows,
)
from compute_structural_subspace import (  # noqa: E402
    coupling_matrix, degree_matched_pairs, materialise,
)
from compute_subspace_divergence import rank_dims as separation_rank  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="/dataMeR1/phil/data")
    ap.add_argument("--graphs", nargs="*", default=None)
    ap.add_argument("--out", default="subspace_overlap.json")
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--pairs", type=int, default=50_000)
    ap.add_argument("--feat-sample", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    names = args.graphs or list(DEFAULT_GRAPHS)
    feats: dict[str, np.ndarray] = {}
    coup_score: dict[str, np.ndarray] = {}

    for name in names:
        path = Path(args.data_root) / DEFAULT_GRAPHS[name]
        if not path.exists():
            log(f"{name}: missing, skipping")
            continue
        obj = load_graph(path)
        x = obj["x"]
        n = int(x.shape[0])
        rows, _i, _m = sample_feature_rows(x, n, args.feat_sample, rng)
        edge_index = as_numpy(obj["edge_index"]).astype(np.int64)
        f = materialise(x, degree_matched_pairs(edge_index, n, args.pairs, rng))
        if f is None or len(rows) < 400:
            log(f"{name}: insufficient sample, skipping")
            continue
        feats[name] = rows
        coup_score[name] = np.diag(coupling_matrix(f))
        log(f"{name}: {len(rows)} bio rows, {len(f['a'])} edge pairs")
        del obj, x, edge_index, f

    D = int(next(iter(feats.values())).shape[1])
    k = args.k
    chance = k * k / D
    out = {"meta": {"k": k, "dims": D, "chance_overlap": chance, "seed": args.seed},
           "pairs": {}}

    log("")
    log(f"top-{k} separation dims vs top-{k} coupling dims (chance overlap {chance:.1f})")
    log(f"  {'pair':44s} {'ovl(A)':>7s} {'ovl(B)':>7s} {'rho(A)':>8s} {'rho(B)':>8s}")
    rows_o, rows_r = [], []
    for a, b in itertools.combinations(sorted(feats), 2):
        sep = separation_rank(feats[a], feats[b])[:k]
        rec = {}
        for g in (a, b):
            cd = np.argsort(-coup_score[g])[:k]
            ov = len(set(sep.tolist()) & set(cd.tolist()))
            rho = stats.spearmanr(
                np.abs(feats[a].mean(0) - feats[b].mean(0))
                / (np.sqrt(0.5 * (feats[a].var(0) + feats[b].var(0))) + 1e-12),
                coup_score[g]).statistic
            rec[g] = {"overlap": ov, "spearman_scores": float(rho)}
            rows_o.append(ov)
            rows_r.append(rho)
        out["pairs"][f"{a}|{b}"] = rec
        log(f"  {a[:20]+' vs '+b[:20]:44s} {rec[a]['overlap']:7d} {rec[b]['overlap']:7d} "
            f"{rec[a]['spearman_scores']:8.3f} {rec[b]['spearman_scores']:8.3f}")

    mo, mr = float(np.mean(rows_o)), float(np.mean(rows_r))
    out["summary"] = {"mean_overlap": mo, "chance": chance,
                      "enrichment": mo / chance, "mean_spearman": mr}
    log("")
    log(f"  mean overlap {mo:.1f} of {k} vs chance {chance:.1f} "
        f"-> {mo/chance:.2f}x enrichment")
    log(f"  mean Spearman(separation score, coupling score) = {mr:+.3f}")
    log(f"  verdict: {'OVERLAPPING - alignment would remove task signal' if mo/chance > 2 or abs(mr) > 0.3 else 'DISJOINT - the cross-graph shift is nuisance'}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, default=float))
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
