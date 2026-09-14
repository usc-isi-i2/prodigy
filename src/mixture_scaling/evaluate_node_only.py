from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

from .binary_metrics import evaluation_report, SCHEMA_VERSION
from .evaluation_artifacts import signature, destination, atomic_json, save_scores
from .config import load_config
from .evaluate_lattice import LP_TARGETS, cached_lp_views, load_pair_module, unwrap
from .lattice import SOURCE_ORDER
from .model import MaskedFeatureMLP, NodeMLP
from .node_only_transfer import feature_batches, feature_split, masked_feature_loss, singleton_rows


FP_TARGETS = SOURCE_ORDER


def load_model(config, checkpoint, objective, device):
    protocol = dict(config["protocol"])
    protocol.update(checkpoint["metadata"].get("protocol", {}))
    args = (
        int(protocol["input_dim"]), int(protocol["hidden_dim"]),
        int(protocol["output_dim"]), float(protocol["dropout"]),
    )
    model = NodeMLP(*args) if objective == "lp" else MaskedFeatureMLP(*args)
    state = checkpoint["model"] if objective == "lp" else checkpoint["pretrain_model"]
    model.load_state_dict(state)
    return model.to(device).eval(), protocol


@torch.no_grad()
def evaluate_fp(target, args, config, rows, device, output_root):
    graph = unwrap(config["graphs"][target]["path"])
    indices = feature_split(
        int(graph.num_nodes), True, args.seed, float(config["protocol"]["node_validation_fraction"])
    )
    for run_id, _ in rows:
        output = output_root / "fp" / f"{run_id}__to__{target}.json"
        if output.is_file():
            continue
        checkpoint = torch.load(
            Path(args.state_root) / "fp" / run_id / "best.pt",
            map_location="cpu", weights_only=False,
        )
        model, protocol = load_model(config, checkpoint, "fp", device)
        replicate_losses = []
        for replicate in range(int(protocol.get("fp_eval_mask_replicates", 10))):
            generator = torch.Generator().manual_seed(args.seed + 80021 + replicate)
            losses, weights = [], []
            for batch in feature_batches(
                graph.x, indices, int(protocol.get("eval_batch_size", 1024)), False, args.seed
            ):
                losses.append(float(masked_feature_loss(
                    model, batch, float(protocol["mask_rate"]), float(protocol["sce_alpha"]),
                    generator, device,
                )))
                weights.append(len(batch))
            replicate_losses.append(float(np.average(losses, weights=weights)))
        payload = {
            "status": "complete", "task": "masked_feature_prediction", "target": target,
            "run_id": run_id, "objective": "fp", "sources": checkpoint["metadata"]["sources"],
            "checkpoint_step": int(checkpoint["step"]), "seed": args.seed,
            "input_view": "center_node_features_only", "mask_kind": "coordinate",
            "mask_rate": float(protocol["mask_rate"]), "n_nodes": len(indices),
            "mask_replicates": len(replicate_losses),
            "scaled_cosine_error_mean": float(np.mean(replicate_losses)),
            "scaled_cosine_error_std": float(np.std(replicate_losses)),
            "replicate_losses": replicate_losses,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n")


@torch.no_grad()
def evaluate_lp(target, args, config, rows, device, output_root, pair):
    graph = unwrap(config["graphs"][target]["path"])
    background_edges, holdout_edges = cached_lp_views(args.state_root, target, args.seed)
    n_nodes = int(graph.num_nodes)
    background = pair.Adjacency.from_edge_index(background_edges.numpy(), n_nodes)
    holdout = pair.Adjacency.from_edge_index(holdout_edges.numpy(), n_nodes)
    rng = np.random.default_rng(args.seed)
    pairs = pair.build_pair_set(
        background, holdout, "degree_matched", rng,
        max_positives=int(config["protocol"].get("lp_eval_positives", 2000)),
        holdout_edge_index=holdout_edges.numpy(),
    )
    val_mask = pair.split_val_mask(pairs, rng)
    nodes = pairs.nodes()
    for run_id, _ in rows:
        primary = output_root / "lp" / f"{run_id}__to__{target}.json"
        primary.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path = Path(args.state_root) / "lp" / run_id / "best.pt"
        provenance = signature(checkpoint_path, config["graphs"][target]["path"],
            Path(args.state_root) / "_cache" / f"{target}_edge_split_s{args.seed}.pt",
            args.seed, int(config["protocol"].get("lp_eval_positives", 2000)))
        with primary.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            output = destination(primary)
            if output.is_file():
                previous = json.loads(output.read_text())
                if previous.get('provenance') != provenance:
                    raise ValueError(f"evaluation provenance changed; use a new output root: {output}")
                if (output.parent / previous['predictions_file']).is_file():
                    continue
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            model, _ = load_model(config, checkpoint, "lp", device)
            table = model(graph.x[nodes].float().to(device)).cpu().numpy()
            embeddings = pair.NodeEmbeddings(table, nodes, n_nodes)
            cosine = pair.pair_scores(embeddings, pairs, "cosine")
            logits = pair.pair_scores(embeddings, pairs, "dot")
            metrics = evaluation_report(pairs.label, logits, val_mask, cosine)
            report = pair.evaluate_scores("node_mlp_cosine", pairs.label, cosine, val_mask).as_dict()
            scores_path = output.with_suffix('.scores.npz')
            save_scores(scores_path, u=pairs.u, v=pairs.v, labels=pairs.label,
                        dot_logits=logits, cosine_scores=cosine, validation_mask=val_mask)
            payload = {
                "schema_version": SCHEMA_VERSION, "status": "complete", "task": "static_link_prediction", "target": target,
                "run_id": run_id, "objective": "lp", "sources": checkpoint["metadata"]["sources"],
                "checkpoint_step": int(checkpoint["step"]), "seed": args.seed,
                "input_view": "center_node_features_only", "edge_partition": "canonical_lp_training_cache",
                "negative_kind": "degree_matched", "n_pairs": len(pairs), "report": report,
                "legacy_report_score_kind": "validation-oriented cosine; hits_at_50 is pooled-pair precision, not per-query retrieval",
                "metrics": metrics, "provenance": provenance, "predictions_file": scores_path.name,
                "gates": {
                    "holdout_leakage_edges": pair.leakage_check(background, pairs),
                    "endpoint_sensitivity": pair.endpoint_sensitivity(embeddings, pairs),
                    "endpoint_permutation_auc": pair.endpoint_permutation_auc(
                        embeddings, pairs, np.random.default_rng(args.seed + 1)),
                },
            }
            if payload['gates']['holdout_leakage_edges'] != 0:
                raise ValueError('evaluation pair leakage gate failed')
            atomic_json(output, payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--objective", choices=("lp", "fp"), required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--state-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--prodigy-root", default="/dataMeR1/phil/gfm/prodigy-nm-pairs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-id")
    parser.add_argument("--target")
    args = parser.parse_args()
    if args.device not in (0, 1, 2, 3):
        raise ValueError("only owned Tucker GPUs 0-3 are allowed")
    config = load_config(args.config)
    rows = singleton_rows()
    if args.run_id:
        rows = [row for row in rows if row[0] == args.run_id]
        if not rows:
            parser.error(f"unknown --run-id {args.run_id!r}")
    targets = FP_TARGETS if args.objective == "fp" else LP_TARGETS
    if args.target:
        if args.target not in targets:
            parser.error(f"invalid target {args.target!r}")
        targets = (args.target,)
    assigned = targets[args.worker_index::args.workers]
    pair = load_pair_module(Path(args.prodigy_root)) if args.objective == "lp" else None
    device = torch.device(f"cuda:{args.device}")
    for target in assigned:
        if args.objective == "fp":
            evaluate_fp(target, args, config, rows, device, Path(args.output_root))
        else:
            evaluate_lp(target, args, config, rows, device, Path(args.output_root), pair)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
