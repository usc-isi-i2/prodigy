#!/usr/bin/env python3
"""Do the graph feature clouds separate if you pick the right dimensions?

``compute_graph_divergence.py`` reports centroid cosine distances of .001-.075
across all 768 GTE dims -- the clouds look nearly coincident. That is an average
over every dimension, so a real separation confined to a few coordinates would be
diluted away. This asks the same question inside a selected subspace.

Two controls decide whether the answer means anything:

  * **Held-out.** Dimensions are ranked on a fit half of each graph's sample and
    every distance is measured on a disjoint eval half. Selecting the k most
    separating dims and then measuring separation in those same rows is circular
    and will manufacture a gap from noise with 768 candidates.
  * **Random-subspace baseline.** Restricting to k dims inflates distances on its
    own: fewer coordinates means less averaging, so cosine distances grow whether
    or not the subspace is meaningful. Every selected-k number is therefore
    reported against k *random* dims (mean over --random-draws) at the same k.
    Selected-vs-random is the real signal; selected-vs-full is not.

Only ``x`` is read -- no edge_index -- so even the 23M-node graphs load fast.
Zero (missing-bio) rows are excluded, matching the parent script, so nothing here
is driven by the differing bio-coverage rates.

    python scripts/experiments/analysis/graph_divergence/compute_subspace_divergence.py \
        --data-root /dataMeR1/phil/data \
        --out scripts/experiments/analysis/graph_divergence/data/subspace_divergence.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_graph_divergence import (  # noqa: E402
    DEFAULT_GRAPHS,
    load_graph,
    log,
    proxy_a_distance,
    sample_feature_rows,
)

KS = (8, 16, 32, 64, 128, 256, 768)


def centroid_cosdist(a: np.ndarray, b: np.ndarray) -> float:
    ma, mb = a.mean(0), b.mean(0)
    na, nb = np.linalg.norm(ma), np.linalg.norm(mb)
    if na == 0 or nb == 0:
        return float("nan")
    return float(1.0 - (ma @ mb) / (na * nb))


def rank_dims(fa: np.ndarray, fb: np.ndarray) -> np.ndarray:
    """Per-dim standardised mean difference, computed on the fit halves only."""
    d = np.abs(fa.mean(0) - fb.mean(0))
    s = np.sqrt(0.5 * (fa.var(0) + fb.var(0))) + 1e-12
    return np.argsort(-(d / s))


def measure(ea: np.ndarray, eb: np.ndarray, dims: np.ndarray,
            rng: np.random.Generator) -> dict[str, float]:
    a, b = ea[:, dims], eb[:, dims]
    return {"centroid_cosdist": centroid_cosdist(a, b),
            "proxy_a": proxy_a_distance(a, b, rng)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="/dataMeR1/phil/data")
    ap.add_argument("--graphs", nargs="*", default=None)
    ap.add_argument("--out", default="subspace_divergence.json")
    ap.add_argument("--feat-sample", type=int, default=8000,
                    help="non-missing rows per graph; split in half fit/eval")
    ap.add_argument("--random-draws", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    names = args.graphs or list(DEFAULT_GRAPHS)
    root = Path(args.data_root)
    rng = np.random.default_rng(args.seed)

    fit: dict[str, np.ndarray] = {}
    ev: dict[str, np.ndarray] = {}
    for name in names:
        path = root / DEFAULT_GRAPHS[name]
        if not path.exists():
            log(f"{name}: missing {path}, skipping")
            continue
        obj = load_graph(path)
        x = obj["x"]
        rows, _idx, miss = sample_feature_rows(x, int(x.shape[0]), args.feat_sample, rng)
        if len(rows) < 400:
            log(f"{name}: only {len(rows)} rows with a bio, skipping")
            continue
        perm = rng.permutation(len(rows))
        half = len(rows) // 2
        fit[name], ev[name] = rows[perm[:half]], rows[perm[half:]]
        log(f"{name}: {len(rows)} bio rows (missing_rate {miss:.3f}), "
            f"fit {len(fit[name])} / eval {len(ev[name])}")
        del obj, x

    D = int(next(iter(ev.values())).shape[1])
    ks = [k for k in KS if k <= D]
    out: dict[str, Any] = {
        "meta": {"feat_sample": args.feat_sample, "random_draws": args.random_draws,
                 "seed": args.seed, "dims": D,
                 "note": "dims ranked on fit half, all distances measured on eval half; "
                         "zero (missing-bio) rows excluded"},
        "pairs": {},
    }

    for a, b in itertools.combinations(sorted(ev), 2):
        order = rank_dims(fit[a], fit[b])
        rec: dict[str, Any] = {"selected": {}, "random": {}}
        for k in ks:
            rec["selected"][str(k)] = measure(ev[a], ev[b], np.sort(order[:k]), rng)
            draws = [measure(ev[a], ev[b], np.sort(rng.choice(D, k, replace=False)), rng)
                     for _ in range(args.random_draws)]
            rec["random"][str(k)] = {
                m: float(np.mean([d[m] for d in draws if d[m] is not None]))
                for m in ("centroid_cosdist", "proxy_a")}
        out["pairs"][f"{a}|{b}"] = rec
        full = rec["selected"][str(D)]
        best = max(ks, key=lambda k: rec["selected"][str(k)]["centroid_cosdist"])
        log(f"{a} vs {b}: full cosdist {full['centroid_cosdist']:.4f} "
            f"(proxy-A {full['proxy_a']:.3f}) | best k={best} selected "
            f"{rec['selected'][str(best)]['centroid_cosdist']:.4f} vs random-{best} "
            f"{rec['random'][str(best)]['centroid_cosdist']:.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, default=float))
    log(f"wrote {args.out}")

    log("")
    log("mean over all pairs -- centroid cosine distance, selected vs random dims")
    log(f"  {'k':>5s} {'selected':>10s} {'random':>10s} {'ratio':>7s} "
        f"{'proxyA sel':>11s} {'proxyA rnd':>11s}")
    for k in ks:
        s = np.mean([p["selected"][str(k)]["centroid_cosdist"] for p in out["pairs"].values()])
        r = np.mean([p["random"][str(k)]["centroid_cosdist"] for p in out["pairs"].values()])
        ps = np.mean([p["selected"][str(k)]["proxy_a"] for p in out["pairs"].values()])
        pr = np.mean([p["random"][str(k)]["proxy_a"] for p in out["pairs"].values()])
        log(f"  {k:5d} {s:10.4f} {r:10.4f} {s/max(r,1e-9):7.2f} {ps:11.3f} {pr:11.3f}")


if __name__ == "__main__":
    main()
