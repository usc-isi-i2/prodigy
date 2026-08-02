"""Score many frozen encoders on one graph with the few-shot regression probe.

Companion to ``regression_probe.py`` (which holds the protocol and self-tests
offline). The loop is inverted the same way ``pair_link_sweep.py`` inverts it: the
graph artifact is 1-78 GB and loading it dominates, so load ONCE, build ONE episode
set per target, and score every checkpoint against it.

Every arm therefore sees identical support and query nodes, and the raw-feature
floor is computed on those same nodes -- the comparison the old episodic eval could
not make.

Only the nodes an episode actually touches are embedded (~episodes x (shots +
n_query) per target), not the whole graph: ~11k nodes instead of covid19's 23M.

    python scripts/eval/regression_probe_sweep.py \\
        --graph /dataMeR1/phil/data/midterm/graphs/retweet_graph_parquet.pt \\
        --dataset midterm --model-list models.txt --out-dir results/ --device cuda

``--no-encoder`` skips checkpoints entirely and reports only the raw-feature floor,
which is what the gate uses to check this implementation against the published
``features_only_floor.csv``.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.regression_probe import (  # noqa: E402
    apply_transform, build_episodes, probe_spearman,
)
from scripts.eval.pair_link_ckpt import (  # noqa: E402
    ENCODER_DEFAULTS, _get_field, build_subgraph_dataset, load_frozen_encoder,
    load_graph_blob,
)
from scripts.eval.pair_link_eval import embeddings_by_node  # noqa: E402

FIELDS = ["dataset", "model", "target", "transform", "shots", "n_query", "episodes",
          "alpha", "features", "n_hop", "hop_sizes", "node_limit", "spearman", "rmse",
          "r2", "n_pred", "n_labeled"]


def parse_model_list(path: str) -> List[Tuple[str, str]]:
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            raise ValueError(f"bad model-list line: {line!r}")
        out.append((parts[0], parts[1]))
    return out


def node_targets(blob, target: str, transform: str):
    """(node_ids, transformed values) for nodes the benchmark counts as labeled.

    Graph artifacts carry the profile panel under ``node_targets`` as a dict of
    name -> tensor aligned to node index. Order matters: the loader
    (``data/midterm.py:538``) transforms FIRST and counts finite entries after, so
    the transform itself can drop rows. Masking before transforming would keep rows
    the benchmark discards.
    """
    nt = _get_field(blob, "node_targets")
    if nt is None or target not in nt:
        return None, None
    y = np.asarray(nt[target], dtype=np.float64).reshape(-1)
    y = apply_transform(y, transform)
    mask = np.isfinite(y)
    return np.flatnonzero(mask), y[mask]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model-list")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--no-encoder", action="store_true",
                    help="raw-feature floor only; no checkpoint needed (the gate).")
    ap.add_argument("--targets", default="followers_count,statuses_count,account_age_days")
    ap.add_argument("--transform", default="log1p", choices=["none", "log1p"],
                    help="Must match the benchmark's --reg-transform to compare.")
    ap.add_argument("--shots", type=int, default=10)
    ap.add_argument("--n-query", type=int, default=12)
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--alpha", default="1.0",
                    help="Comma-separated ridge alphas; one row per alpha.")
    ap.add_argument("--background-view", default="static_background")
    ap.add_argument("--n-hop", type=int, default=1)
    ap.add_argument("--hop-sizes", default="",
                    help="Optional comma-separated fanout per extracted hop.")
    ap.add_argument("--node-limit", type=int, default=2000)
    ap.add_argument("--emb-dim", type=int, default=256)
    ap.add_argument("--input-dim", type=int, default=768)
    ap.add_argument("--gnn-type", default="sage")
    ap.add_argument("--n-layer", type=int, default=1)
    ap.add_argument("--layers", default="S,U,M")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    from experiments.sampler import parse_hop_sizes
    hop_sizes = parse_hop_sizes(args.hop_sizes, args.n_hop)
    provenance = dict(n_hop=args.n_hop, hop_sizes=args.hop_sizes,
                      node_limit=args.node_limit)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{args.dataset}__reg_probe.csv"

    models = parse_model_list(args.model_list) if args.model_list and not args.no_encoder else []
    alphas = [float(a) for a in args.alpha.split(",") if a.strip()]
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]

    t0 = time.time()
    print(f"[{args.dataset}] loading graph artifact ...", flush=True)
    blob, graph = load_graph_blob(args.graph)
    n_nodes = int(graph.num_nodes)
    raw_x = _get_field(blob, "x")
    print(f"[{args.dataset}] nodes={n_nodes} ({time.time()-t0:.0f}s)", flush=True)

    rows = []
    subgraph_ds = None

    for target in targets:
        node_ids, y = node_targets(blob, target, args.transform)
        if node_ids is None or len(node_ids) == 0:
            print(f"[{args.dataset}] skip {target}: no labeled nodes", flush=True)
            continue

        ep = build_episodes(node_ids, y, args.shots, args.n_query, args.episodes,
                            seed=args.seed)
        if ep is None:
            print(f"[{args.dataset}] skip {target}: too few labeled nodes", flush=True)
            continue
        print(f"[{args.dataset}] {target}: {len(node_ids)} labeled, "
              f"{args.episodes} episodes, {len(ep.nodes)} distinct nodes", flush=True)

        # ---- raw-feature floor, on the SAME episodes ------------------------
        feats_raw = np.asarray(raw_x[ep.nodes], dtype=np.float64)
        for alpha in alphas:
            res = probe_spearman(feats_raw, ep, alpha=alpha)
            rows.append(dict(dataset=args.dataset, model="__features_only__",
                             target=target, transform=args.transform,
                             shots=args.shots, n_query=args.n_query,
                             episodes=args.episodes, features="raw_x",
                             n_labeled=int(len(node_ids)), **provenance, **res))
            print(f"  [floor] raw_x alpha={alpha}: rho={res['spearman']:+.4f}", flush=True)

        # ---- frozen encoders ------------------------------------------------
        if models and subgraph_ds is None:
            subgraph_ds = build_subgraph_dataset(
                blob, graph, args.n_hop, args.background_view,
                hop_sizes=hop_sizes, node_limit=args.node_limit,
            )
        for name, ckpt in models:
            params = dict(ENCODER_DEFAULTS)
            params.update(emb_dim=args.emb_dim, input_dim=args.input_dim,
                          gnn_type=args.gnn_type, n_layer=args.n_layer,
                          layers=args.layers)
            model = load_frozen_encoder(ckpt, params, device=args.device)
            emb = embeddings_by_node(model, subgraph_ds, np.asarray(ep.nodes),
                                     n_nodes, batch_size=args.batch_size,
                                     device=args.device)
            feats = np.asarray(emb[np.asarray(ep.nodes)], dtype=np.float64)
            for alpha in alphas:
                res = probe_spearman(feats, ep, alpha=alpha)
                rows.append(dict(dataset=args.dataset, model=name, target=target,
                                 transform=args.transform, shots=args.shots,
                                 n_query=args.n_query, episodes=args.episodes,
                                 features="frozen_emb",
                                 n_labeled=int(len(node_ids)), **provenance, **res))
                print(f"  {name} alpha={alpha}: rho={res['spearman']:+.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)

        # stream after each target so a long sweep is never lost
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)

    with (out_dir / f"{args.dataset}__reg_probe.json").open("w", encoding="utf-8") as fh:
        json.dump({"dataset": args.dataset, "graph": args.graph, "rows": rows,
                   "shots": args.shots, "n_query": args.n_query,
                   "episodes": args.episodes, "transform": args.transform,
                   "seed": args.seed}, fh, indent=2)
    print(f"[{args.dataset}] wrote {csv_path} ({len(rows)} rows, "
          f"{time.time()-t0:.0f}s total)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
