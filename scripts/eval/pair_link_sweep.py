"""Rescore many frozen encoders on one graph with the pair-conditioned evaluator.

The graph artifacts are 1-78GB and the adjacency build is the dominant cost, so
the loop is inverted relative to the old eval: load the graph ONCE, then score
every checkpoint against it.

Critically, the pair set is built once and **shared across all models**. Every arm
is therefore scored on identical positives and identical negatives, which the old
per-run episodic sampling did not guarantee. Heuristic and raw-feature floors are
computed once on that same pair set.

Results stream to CSV as each model finishes, so a long sweep is never lost.

    python scripts/eval/pair_link_sweep.py \\
        --graph /dataMeR1/phil/data/midterm/graphs/retweet_graph_parquet.pt \\
        --dataset midterm --model-list models.txt --out-dir results/ --device cuda

``model-list`` is one ``name<TAB or space>checkpoint_path`` per line.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.pair_link_eval import (  # noqa: E402
    Adjacency, HEURISTICS, NodeEmbeddings, PairSet, build_pair_set,
    embeddings_by_node, endpoint_permutation_auc, endpoint_sensitivity,
    evaluate_scores, heuristic_scores, leakage_check, lock_decision_threshold,
    pair_scores, split_val_mask,
)
from scripts.eval.pair_link_ckpt import (  # noqa: E402
    ENCODER_DEFAULTS, _get_field, _view_edge_index, build_subgraph_dataset,
    load_frozen_encoder, load_graph_blob,
)

FIELDS = ["dataset", "model", "negative_kind", "scorer", "auc", "average_precision",
          "hits_at_50", "orientation", "n_pairs", "n_positive",
          "endpoint_permutation_auc", "endpoint_sensitivity", "leakage_edges"]


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model-list", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--background-view", default="static_background")
    ap.add_argument("--holdout-view", default="static_holdout")
    ap.add_argument("--negative-kinds", default="degree_matched,random,hard_2hop")
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
    ap.add_argument(
        "--export-examples",
        action="store_true",
        help="Write validation-locked test-pair predictions to <dataset>__pair_lp_examples.jsonl.",
    )
    ap.add_argument("--context-neighbors", type=int, default=3)
    args = ap.parse_args()

    import torch

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{args.dataset}__pair_lp.csv"
    json_path = out_dir / f"{args.dataset}__pair_lp.json"
    examples_path = out_dir / f"{args.dataset}__pair_lp_examples.jsonl"
    if args.export_examples and examples_path.exists():
        examples_path.unlink()

    models = parse_model_list(args.model_list)
    kinds = [k.strip() for k in args.negative_kinds.split(",")]

    t0 = time.time()
    print(f"[{args.dataset}] loading graph artifact ...", flush=True)
    blob, graph = load_graph_blob(args.graph)
    n = int(graph.num_nodes)
    bg_ei = np.asarray(_view_edge_index(blob, args.background_view))
    ho_ei = np.asarray(_view_edge_index(blob, args.holdout_view))
    print(f"[{args.dataset}] nodes={n} bg_edges={bg_ei.shape[1]} "
          f"holdout_edges={ho_ei.shape[1]} ({time.time()-t0:.0f}s)", flush=True)

    print(f"[{args.dataset}] building adjacency ...", flush=True)
    background = Adjacency.from_edge_index(bg_ei, n)
    holdout = Adjacency.from_edge_index(ho_ei, n)
    print(f"[{args.dataset}] adjacency ready ({time.time()-t0:.0f}s)", flush=True)

    # ---- ONE pair set per negative kind, shared by every model ----------------
    pair_sets: Dict[str, PairSet] = {}
    val_masks: Dict[str, np.ndarray] = {}
    for kind in kinds:
        rng = np.random.default_rng(args.seed)
        ps = build_pair_set(background, holdout, kind, rng,
                            max_positives=args.max_positives,
                            holdout_edge_index=ho_ei)
        pair_sets[kind] = ps
        val_masks[kind] = split_val_mask(ps, rng)
        print(f"[{args.dataset}] {kind}: {len(ps)} pairs "
              f"({int((ps.label==1).sum())} positive), leakage={leakage_check(background, ps)}",
              flush=True)

    all_nodes = np.unique(np.concatenate([ps.nodes() for ps in pair_sets.values()]))
    print(f"[{args.dataset}] {all_nodes.size} distinct nodes to embed per model", flush=True)

    rows: List[dict] = []

    def emit(row: dict) -> None:
        rows.append(row)
        write_header = not csv_path.exists()
        with csv_path.open("a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            if write_header:
                w.writeheader()
            w.writerow(row)

    # ---- floors: computed once on the shared pair set -------------------------
    raw_x = _get_field(blob, "x")
    if raw_x is None:
        raw_x = getattr(graph, "x", None)
    raw_feats = (NodeEmbeddings(np.asarray(raw_x[torch.as_tensor(all_nodes)],
                                           dtype=np.float32), all_nodes, n)
                 if raw_x is not None else None)

    for kind in kinds:
        ps, vm = pair_sets[kind], val_masks[kind]
        leak = float(leakage_check(background, ps))
        for h in HEURISTICS:
            r = evaluate_scores(h, ps.label, heuristic_scores(h, ps, background), vm)
            emit({"dataset": args.dataset, "model": "__floor__", "negative_kind": kind,
                  "scorer": h, "auc": r.auc, "average_precision": r.average_precision,
                  "hits_at_50": r.hits_at_50, "orientation": r.orientation,
                  "n_pairs": r.n_pairs, "n_positive": r.n_positive,
                  "endpoint_permutation_auc": "", "endpoint_sensitivity": "",
                  "leakage_edges": leak})
        if raw_feats is not None:
            r = evaluate_scores("raw_feature_cosine", ps.label,
                                pair_scores(raw_feats, ps, "cosine"), vm)
            emit({"dataset": args.dataset, "model": "__floor__", "negative_kind": kind,
                  "scorer": "raw_feature_cosine", "auc": r.auc,
                  "average_precision": r.average_precision, "hits_at_50": r.hits_at_50,
                  "orientation": r.orientation, "n_pairs": r.n_pairs,
                  "n_positive": r.n_positive, "endpoint_permutation_auc": "",
                  "endpoint_sensitivity": "", "leakage_edges": leak})
    print(f"[{args.dataset}] floors done ({time.time()-t0:.0f}s)", flush=True)

    dataset_obj = build_subgraph_dataset(blob, graph, args.n_hop, args.background_view)

    params = dict(ENCODER_DEFAULTS)
    params.update(emb_dim=args.emb_dim, input_dim=args.input_dim,
                  gnn_type=args.gnn_type, n_layer=args.n_layer, layers=args.layers)

    for mi, (name, ckpt) in enumerate(models, 1):
        tm = time.time()
        try:
            model = load_frozen_encoder(ckpt, params, device=args.device,
                                        strict_report=False)
            embedded = embeddings_by_node(
                model, dataset_obj, all_nodes, n,
                device=args.device, batch_size=args.batch_size,
                return_context=args.export_examples,
                context_size=args.context_neighbors,
            )
            if args.export_examples:
                emb, contexts = embedded
            else:
                emb, contexts = embedded, {}
        except Exception as exc:  # keep the sweep alive; a failed arm is reported
            print(f"[{args.dataset}] {name}: FAILED -- {type(exc).__name__}: {exc}",
                  flush=True)
            continue

        for kind in kinds:
            ps, vm = pair_sets[kind], val_masks[kind]
            scores = pair_scores(emb, ps, "cosine")
            r = evaluate_scores("encoder_cosine", ps.label, scores, vm)
            emit({
                "dataset": args.dataset, "model": name, "negative_kind": kind,
                "scorer": "encoder_cosine", "auc": r.auc,
                "average_precision": r.average_precision, "hits_at_50": r.hits_at_50,
                "orientation": r.orientation, "n_pairs": r.n_pairs,
                "n_positive": r.n_positive,
                "endpoint_permutation_auc": endpoint_permutation_auc(
                    emb, ps, np.random.default_rng(args.seed + 1)),
                "endpoint_sensitivity": endpoint_sensitivity(emb, ps),
                "leakage_edges": float(leakage_check(background, ps)),
            })
            if args.export_examples:
                threshold, val_balanced_accuracy = lock_decision_threshold(
                    ps.label, scores, vm, r.orientation
                )
                oriented = scores * int(r.orientation)
                predictions = (oriented >= threshold).astype(np.int8)
                test_indices = np.flatnonzero(~vm)
                with examples_path.open("a", encoding="utf-8") as handle:
                    for pair_index in test_indices.tolist():
                        u = int(ps.u[pair_index])
                        v = int(ps.v[pair_index])
                        gt = int(ps.label[pair_index])
                        pred = int(predictions[pair_index])
                        common = int(np.intersect1d(
                            background.neighbors(u), background.neighbors(v),
                            assume_unique=True,
                        ).size)
                        payload = {
                            "schema_version": 1,
                            "task": "static_link_prediction",
                            "dataset": args.dataset,
                            "split": "test",
                            "model": name,
                            "negative_kind": kind,
                            "pair_index": int(pair_index),
                            "u": u,
                            "v": v,
                            "u_context_node_ids": contexts.get(u, []),
                            "v_context_node_ids": contexts.get(v, []),
                            "gt": gt,
                            "prediction": pred,
                            "correct": bool(gt == pred),
                            "error_type": ("tp" if gt and pred else
                                           "fn" if gt and not pred else
                                           "fp" if not gt and pred else "tn"),
                            "raw_score": float(scores[pair_index]),
                            "oriented_score": float(oriented[pair_index]),
                            "orientation": int(r.orientation),
                            "decision_threshold": float(threshold),
                            "validation_balanced_accuracy": float(val_balanced_accuracy),
                            "u_degree": int(background.degree[u]),
                            "v_degree": int(background.degree[v]),
                            "common_neighbors": common,
                        }
                        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        best = [r for r in rows if r["model"] == name and r["negative_kind"] == kinds[0]]
        auc = best[0]["auc"] if best else float("nan")
        print(f"[{args.dataset}] ({mi}/{len(models)}) {name}: "
              f"{kinds[0]} AUC={auc:.3f}  ({time.time()-tm:.0f}s)", flush=True)

        del model, emb

    json_path.write_text(json.dumps(rows, indent=2))
    print(f"[{args.dataset}] wrote {csv_path} and {json_path} "
          f"(total {time.time()-t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
