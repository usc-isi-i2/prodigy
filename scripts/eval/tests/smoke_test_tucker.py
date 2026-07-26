"""Tucker-side smoke test: real checkpoint -> encoder -> embeddings -> pair scores.

Run this BEFORE any full rescoring sweep. It validates the one thing that cannot
be checked on a laptop: that ``build_encoder`` reconstructs the trained
architecture well enough for ``state_dict['model']`` to load with no *encoder*
weights missing. A silent mismatch there would score with partly-random weights.

    python scripts/eval/tests/smoke_test_tucker.py \\
        --graph /dataMeR1/phil/data/covid19_twitter/graphs/retweet_graph_parquet.pt \\
        --checkpoint /dataMeR1/phil/gfm/prodigy-mtr/state/mtr_MIX_.../checkpoint/state_dict_30000.ckpt \\
        --device cuda

Exits non-zero on any failure. Cheap: embeds a few hundred nodes only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from scripts.eval.pair_link_eval import (  # noqa: E402
    Adjacency, build_pair_set, embeddings_by_node, evaluate_graph,
)
from scripts.eval.pair_link_ckpt import (  # noqa: E402
    ENCODER_DEFAULTS, build_subgraph_dataset, load_frozen_encoder, _view_edge_index,
    load_graph_blob,
)

OK = True


def check(name: str, cond: bool, detail: str = "") -> None:
    global OK
    OK &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--emb-dim", type=int, default=256)
    ap.add_argument("--input-dim", type=int, default=768)
    ap.add_argument("--gnn-type", default="sage")
    ap.add_argument("--n-layer", type=int, default=1)
    ap.add_argument("--layers", default="S,U,M")
    ap.add_argument("--n-hop", type=int, default=1)
    ap.add_argument("--max-positives", type=int, default=150)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch

    print("Tucker smoke test")
    blob, graph = load_graph_blob(args.graph)
    n = int(graph.num_nodes)

    bg_ei = np.asarray(_view_edge_index(blob, "static_background"))
    ho_ei = np.asarray(_view_edge_index(blob, "static_holdout"))
    print(f"  graph: nodes={n} bg_edges={bg_ei.shape[1]} holdout_edges={ho_ei.shape[1]}")
    print("  building adjacency (this dominates runtime on the big graphs)...")
    background = Adjacency.from_edge_index(bg_ei, n)
    holdout = Adjacency.from_edge_index(ho_ei, n)

    params = dict(ENCODER_DEFAULTS)
    params.update(emb_dim=args.emb_dim, input_dim=args.input_dim,
                  gnn_type=args.gnn_type, n_layer=args.n_layer, layers=args.layers)

    # load_frozen_encoder raises if any encoder weight is missing
    model = load_frozen_encoder(args.checkpoint, params, device=args.device)
    check("checkpoint loads with all encoder weights present", True)

    pairs = build_pair_set(background, holdout, "degree_matched",
                           np.random.default_rng(0), max_positives=args.max_positives,
                           holdout_edge_index=ho_ei)
    check("scored positives are absent from the background graph",
          int(background.contains_pairs(pairs.u[pairs.label == 1],
                                        pairs.v[pairs.label == 1]).sum()) == 0)

    nodes = pairs.nodes()
    print(f"  embedding {nodes.size} nodes")

    dataset = build_subgraph_dataset(blob, graph, args.n_hop, "static_background")
    emb = embeddings_by_node(model, dataset, nodes, n, device=args.device, batch_size=128)

    check("embeddings are finite", bool(np.isfinite(emb[nodes]).all()))
    spread = float(np.std(emb[nodes], axis=0).mean())
    check("embeddings are not collapsed", spread > 1e-6, f"mean_std={spread:.5f}")
    check("embedding dim matches emb_dim", emb.shape[1] == args.emb_dim,
          f"{emb.shape[1]} vs {args.emb_dim}")

    res = evaluate_graph(background, holdout, emb, None, "degree_matched",
                         seed=0, max_positives=args.max_positives,
                         holdout_edge_index=ho_ei)
    g = res["gates"]
    check("encoder scoring is endpoint-sensitive", g["endpoint_sensitivity"] > 0.99,
          f"{g['endpoint_sensitivity']:.3f}")
    check("endpoint permutation collapses the signal",
          abs(g["endpoint_permutation_auc"] - 0.5) < 0.15,
          f"AUC={g['endpoint_permutation_auc']:.3f}")

    print("\n  scores (small sample -- indicative only, not a result):")
    for r in res["reports"]:
        print(f"    {r['name']:26s} AUC={r['auc']:.3f} orient={r['orientation']:+d}")

    print(f"\n{'SMOKE TEST PASSED' if OK else 'SMOKE TEST FAILED'}")
    return 0 if OK else 1


if __name__ == "__main__":
    raise SystemExit(main())
