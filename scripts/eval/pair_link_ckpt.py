"""Load a frozen prodigy encoder and score graphs with the pair-conditioned evaluator.

Companion to ``pair_link_eval.py``: that module holds the protocol and is
checkpoint-free (so it can self-test offline); this one rebuilds a trained
encoder and drives it over real graphs.

Model reconstruction mirrors ``experiments/trainer.py`` exactly -- same
``get_module_list`` call, same ``SingleLayerGeneralGNN`` assembly -- so a
checkpoint's ``state_dict["model"]`` loads without surgery. Only encoder weights
matter here: the label/prototype path is never used, because the pair score is a
function of the two endpoint embeddings alone.

Usage
-----
Heuristic floors only (no checkpoint, runs anywhere)::

    python scripts/eval/pair_link_ckpt.py --graph <graph.pt> --no-encoder

Rescore a frozen checkpoint::

    python scripts/eval/pair_link_ckpt.py \\
        --graph /dataMeR1/phil/data/covid19_twitter/graphs/retweet_graph_parquet.pt \\
        --checkpoint /dataMeR1/phil/gfm/prodigy-mtr/state/mtr_MIX_*/checkpoint/state_dict_30000.ckpt \\
        --emb-dim 256 --gnn-type sage --n-layer 1 --input-dim 768 \\
        --out results/mix_covid.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.pair_link_eval import (  # noqa: E402
    Adjacency,
    evaluate_graph,
    embeddings_by_node,
    build_pair_set,
)


# Defaults that reproduce the trainer's construction for the retweet-graph runs
# (layers "S,U,M", 1-layer SAGE, 256-dim). Anything a run overrode must be passed
# explicitly -- a mismatch surfaces as missing/unexpected state_dict keys, which
# this loader reports rather than silently swallowing.
ENCODER_DEFAULTS = dict(
    layers="S,U,M",
    emb_dim=256,
    input_dim=768,
    gnn_type="sage",
    n_layer=1,
    dropout=0.0,
    reset_after_layer=None,
    attention_mask_scheme=None,
    has_final_back=False,
    msg_pos_only=False,
    batch_norm_metagraph=True,
    batch_norm_encoder=True,
    skip_path=False,
    text_features_dropout=0.0,
    ignore_label_embeddings=True,
    zero_label_embeddings=False,
    zero_shot=True,
    edge_attr_dim=None,
)


def build_encoder(params: dict, device: str = "cpu"):
    """Rebuild the SingleLayerGeneralGNN exactly as the trainer does."""
    import torch
    from experiments.layers import get_module_list
    from models.general_gnn import SingleLayerGeneralGNN

    bert_dim = 768
    layer_list = get_module_list(
        params["layers"],
        params["emb_dim"],
        edge_attr_dim=params["edge_attr_dim"],
        input_dim=params["input_dim"],
        dropout=params["dropout"],
        reset_after_layer=params["reset_after_layer"],
        attention_mask_scheme=params["attention_mask_scheme"],
        has_final_back=params["has_final_back"],
        msg_pos_only=params["msg_pos_only"],
        batch_norm_metagraph=params["batch_norm_metagraph"],
        batch_norm_encoder=params["batch_norm_encoder"],
        encoder_gnn_type=params["gnn_type"],
    )
    model = SingleLayerGeneralGNN(
        layer_list=torch.nn.ModuleList(layer_list),
        initial_label_mlp=torch.nn.Linear(bert_dim, params["emb_dim"]),
        params=params,
        text_dropout=torch.nn.Dropout(params["text_features_dropout"]),
    )
    return model.to(device)


def load_frozen_encoder(checkpoint: str, params: dict, device: str = "cpu",
                        strict_report: bool = True):
    """Build the encoder and load ``state_dict['model']`` from a checkpoint.

    Reports missing/unexpected keys instead of hiding them: a silent mismatch here
    would mean scoring with partly-random weights, which is exactly the class of
    bug this whole effort exists to remove.
    """
    import torch

    model = build_encoder(params, device=device)
    blob = torch.load(checkpoint, map_location=device, weights_only=False)
    if "model" not in blob:
        raise KeyError(
            f"checkpoint {checkpoint} has no 'model' entry; found {sorted(blob)}")
    result = model.load_state_dict(blob["model"], strict=False)

    missing = [k for k in result.missing_keys if not k.startswith("learned_label_embedding")]
    unexpected = list(result.unexpected_keys)
    if strict_report and (missing or unexpected):
        print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")
        for k in missing[:10]:
            print(f"[load]   missing: {k}")
        for k in unexpected[:10]:
            print(f"[load]   unexpected: {k}")
    encoder_missing = [k for k in missing if k.startswith(("layer_list", "initial_input_mlp"))]
    if encoder_missing:
        raise RuntimeError(
            f"{len(encoder_missing)} ENCODER weights missing from the checkpoint "
            f"(e.g. {encoder_missing[:3]}). The --emb-dim/--gnn-type/--n-layer/"
            "--layers flags almost certainly do not match how this run was trained; "
            "scoring would use randomly-initialised weights.")
    model.eval()
    return model


def build_subgraph_dataset(graph, n_hop: int, background_view: str):
    """SubgraphDataset over the background view only (held-out edges excluded)."""
    from data.dataset import SubgraphDataset
    from experiments.sampler import NeighborSampler

    bg = _view_edge_index(graph, background_view)
    scoped = graph.clone()
    scoped.edge_index = bg
    sampler = NeighborSampler(scoped, num_hops=n_hop)
    return SubgraphDataset(scoped, sampler, bidirectional=False)


def _view_edge_index(graph, name: str):
    views = getattr(graph, "edge_index_views", None)
    if isinstance(views, dict) and name in views:
        return views[name]
    legacy = f"edge_index_{name}"
    if hasattr(graph, legacy):
        return getattr(graph, legacy)
    raise KeyError(
        f"edge view {name!r} not found; available: "
        f"{sorted(views) if isinstance(views, dict) else 'none'}. Run "
        "scripts/graph_construction/enrich_all_graphs.sh to add the static views.")


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--checkpoint")
    ap.add_argument("--no-encoder", action="store_true",
                    help="skip the encoder; report heuristic + raw-feature floors only")
    ap.add_argument("--model-name", default="encoder")
    ap.add_argument("--background-view", default="static_background")
    ap.add_argument("--holdout-view", default="static_holdout")
    ap.add_argument("--negative-kinds", default="degree_matched,random,hard_2hop")
    ap.add_argument("--score-kind", default="cosine", choices=("cosine", "dot"))
    ap.add_argument("--max-positives", type=int, default=2000)
    ap.add_argument("--n-hop", type=int, default=1)
    ap.add_argument("--emb-dim", type=int, default=256)
    ap.add_argument("--input-dim", type=int, default=768)
    ap.add_argument("--gnn-type", default="sage")
    ap.add_argument("--n-layer", type=int, default=1)
    ap.add_argument("--layers", default="S,U,M")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out")
    args = ap.parse_args(argv)

    import torch

    raw = torch.load(args.graph, map_location="cpu", weights_only=False)
    graph = raw["graph"] if isinstance(raw, dict) and "graph" in raw else raw
    n = int(graph.num_nodes)

    background = Adjacency.from_edge_index(
        np.asarray(_view_edge_index(graph, args.background_view)), n)
    holdout = Adjacency.from_edge_index(
        np.asarray(_view_edge_index(graph, args.holdout_view)), n)
    print(f"[graph] nodes={n} background_edges={background.indices.size // 2} "
          f"holdout_edges={holdout.indices.size // 2}")

    raw_features = np.asarray(graph.x, dtype=np.float32) if getattr(graph, "x", None) is not None else None

    embeddings = None
    if not args.no_encoder:
        if not args.checkpoint:
            ap.error("--checkpoint is required unless --no-encoder is given")
        params = dict(ENCODER_DEFAULTS)
        params.update(emb_dim=args.emb_dim, input_dim=args.input_dim,
                      gnn_type=args.gnn_type, n_layer=args.n_layer, layers=args.layers)
        model = load_frozen_encoder(args.checkpoint, params, device=args.device)

        # Only nodes that appear in some scored pair need embedding -- with a
        # 2000-positive cap that is a few thousand subgraphs, not the whole graph.
        probe_rng = np.random.default_rng(args.seed)
        needed = set()
        for kind in args.negative_kinds.split(","):
            ps = build_pair_set(background, holdout, kind.strip(),
                                np.random.default_rng(args.seed),
                                max_positives=args.max_positives)
            needed.update(ps.nodes().tolist())
        nodes = np.array(sorted(needed), dtype=np.int64)
        print(f"[embed] embedding {nodes.size} nodes on the background view")

        dataset = build_subgraph_dataset(graph, args.n_hop, args.background_view)
        embeddings = embeddings_by_node(
            model, dataset, nodes, n, device=args.device, batch_size=args.batch_size)

    results = []
    for kind in args.negative_kinds.split(","):
        res = evaluate_graph(background, holdout, embeddings, raw_features,
                             kind.strip(), seed=args.seed,
                             max_positives=args.max_positives,
                             score_kind=args.score_kind)
        results.append(res)
        gates = res["gates"]
        print(f"\n[{kind.strip()}] pairs={res['n_pairs']} "
              f"leakage={int(gates['holdout_leakage_edges'])} "
              f"perm_auc={gates.get('endpoint_permutation_auc', float('nan')):.3f} "
              f"sensitivity={gates.get('endpoint_sensitivity', float('nan')):.3f}")
        for r in res["reports"]:
            print(f"    {r['name']:26s} AUC={r['auc']:.3f}  AP={r['average_precision']:.3f}"
                  f"  orient={r['orientation']:+d}")

    payload = {
        "graph": args.graph,
        "checkpoint": args.checkpoint,
        "model_name": args.model_name,
        "results": results,
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
