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
    evaluate_scores, heuristic_scores, leakage_check, pair_scores, split_val_mask,
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
    ap.add_argument(
        "--hop-sizes",
        default="",
        help=(
            "Comma-separated NeighborSampler fanouts. Empty preserves the sampler "
            "default; fair-two-hop experiments must pass 9,9."
        ),
    )
    ap.add_argument("--node-limit", type=int, default=2000)
    ap.add_argument("--emb-dim", type=int, default=256)
    ap.add_argument("--input-dim", type=int, default=768)
    ap.add_argument("--gnn-type", default="sage")
    ap.add_argument("--n-layer", type=int, default=1)
    ap.add_argument("--layers", default="S,U,M")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Keep complete model rows already present in the dataset CSV and score only missing/invalid models.",
    )
    args = ap.parse_args()

    hop_sizes = (
        [int(value.strip()) for value in args.hop_sizes.split(",") if value.strip()]
        if args.hop_sizes
        else None
    )
    if hop_sizes is not None and len(hop_sizes) != args.n_hop:
        ap.error(
            f"--hop-sizes has {len(hop_sizes)} values but --n-hop is {args.n_hop}"
        )

    import torch

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{args.dataset}__pair_lp.csv"
    json_path = out_dir / f"{args.dataset}__pair_lp.json"

    models = parse_model_list(args.model_list)
    kinds = [k.strip() for k in args.negative_kinds.split(",")]

    existing_rows: List[dict] = []
    completed_models: set[str] = set()
    existing_floor_keys: set[tuple[str, str]] = set()
    if args.resume and csv_path.is_file():
        with csv_path.open(newline="") as fh:
            existing_rows = list(csv.DictReader(fh))
        valid_kinds_by_model: Dict[str, set[str]] = {}
        for row in existing_rows:
            kind = row.get("negative_kind", "")
            if row.get("model") == "__floor__":
                existing_floor_keys.add((kind, row.get("scorer", "")))
                continue
            if row.get("scorer") != "encoder_cosine" or kind not in kinds:
                continue
            try:
                leak = float(row["leakage_edges"])
                sensitivity = float(row["endpoint_sensitivity"])
                permutation = float(row["endpoint_permutation_auc"])
            except (KeyError, TypeError, ValueError):
                continue
            if leak == 0 and sensitivity >= 0.99 and abs(permutation - 0.5) < 0.05:
                valid_kinds_by_model.setdefault(row.get("model", ""), set()).add(kind)
        completed_models = {
            model for model, present in valid_kinds_by_model.items() if set(kinds) <= present
        }
        before = len(models)
        models = [item for item in models if item[0] not in completed_models]
        # Drop partial/invalid rows for models that will be rescored. Otherwise an
        # append-only retry would leave two degree-matched rows and make the result
        # ambiguous. Complete models and the shared floors remain byte-for-byte.
        remaining_names = {name for name, _ in models}
        existing_rows = [
            row for row in existing_rows
            if row.get("model") == "__floor__" or row.get("model") not in remaining_names
        ]
        # A retry of an older append-only sweep may already contain duplicates.
        # Keep the last row for each semantic key before writing the clean resume base.
        deduplicated: Dict[tuple[str, str, str], dict] = {}
        for row in existing_rows:
            deduplicated[(
                row.get("model", ""),
                row.get("negative_kind", ""),
                row.get("scorer", ""),
            )] = row
        existing_rows = list(deduplicated.values())
        existing_floor_keys = {
            (row.get("negative_kind", ""), row.get("scorer", ""))
            for row in existing_rows if row.get("model") == "__floor__"
        }
        with csv_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(existing_rows)
        print(
            f"[{args.dataset}] resume: {before - len(models)} complete, "
            f"{len(models)} model(s) remaining",
            flush=True,
        )

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

    rows: List[dict] = list(existing_rows)

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
            if (kind, h) in existing_floor_keys:
                continue
            r = evaluate_scores(h, ps.label, heuristic_scores(h, ps, background), vm)
            emit({"dataset": args.dataset, "model": "__floor__", "negative_kind": kind,
                  "scorer": h, "auc": r.auc, "average_precision": r.average_precision,
                  "hits_at_50": r.hits_at_50, "orientation": r.orientation,
                  "n_pairs": r.n_pairs, "n_positive": r.n_positive,
                  "endpoint_permutation_auc": "", "endpoint_sensitivity": "",
                  "leakage_edges": leak})
        if raw_feats is not None:
            if (kind, "raw_feature_cosine") in existing_floor_keys:
                continue
            r = evaluate_scores("raw_feature_cosine", ps.label,
                                pair_scores(raw_feats, ps, "cosine"), vm)
            emit({"dataset": args.dataset, "model": "__floor__", "negative_kind": kind,
                  "scorer": "raw_feature_cosine", "auc": r.auc,
                  "average_precision": r.average_precision, "hits_at_50": r.hits_at_50,
                  "orientation": r.orientation, "n_pairs": r.n_pairs,
                  "n_positive": r.n_positive, "endpoint_permutation_auc": "",
                  "endpoint_sensitivity": "", "leakage_edges": leak})
    print(f"[{args.dataset}] floors done ({time.time()-t0:.0f}s)", flush=True)

    dataset_obj = build_subgraph_dataset(
        blob,
        graph,
        args.n_hop,
        args.background_view,
        hop_sizes=hop_sizes,
        node_limit=args.node_limit,
    )

    params = dict(ENCODER_DEFAULTS)
    params.update(emb_dim=args.emb_dim, input_dim=args.input_dim,
                  gnn_type=args.gnn_type, n_layer=args.n_layer, layers=args.layers)

    for mi, (name, ckpt) in enumerate(models, 1):
        tm = time.time()
        try:
            model = load_frozen_encoder(ckpt, params, device=args.device,
                                        strict_report=False)
            emb = embeddings_by_node(model, dataset_obj, all_nodes, n,
                                     device=args.device, batch_size=args.batch_size)
        except Exception as exc:  # keep the sweep alive; a failed arm is reported
            print(f"[{args.dataset}] {name}: FAILED -- {type(exc).__name__}: {exc}",
                  flush=True)
            continue

        for kind in kinds:
            ps, vm = pair_sets[kind], val_masks[kind]
            r = evaluate_scores("encoder_cosine", ps.label,
                                pair_scores(emb, ps, "cosine"), vm)
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
