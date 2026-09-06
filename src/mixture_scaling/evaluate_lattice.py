from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from .config import load_config
from .evaluate import evaluate_embeddings
from .lattice import lattice_rows
from .model import GraphSAGE
from .train import configure_sampling_backend


CLS_TARGETS = (
    "covid_political", "election2020", "facebook_page_reference",
    "twibot20", "ukr_rus_suspended",
)
LP_TARGETS = (
    "ukr_rus_twitter", "covid19_twitter", "midterm",
    "twibot20", "cp_hk_twitter", "facebook_page_reference",
)


def unwrap(path: str | Path):
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "graph" in raw:
        return raw["graph"]
    if isinstance(raw, dict):
        return Data(**raw)
    return raw


def view(graph, name: str) -> torch.Tensor:
    views = getattr(graph, "edge_index_views", None)
    if isinstance(views, dict) and name in views:
        return views[name].long()
    legacy = getattr(graph, f"edge_index_{name}", None)
    if legacy is not None:
        return legacy.long()
    raise KeyError(f"missing required {name} edge view")


def cached_lp_views(state_root: str | Path, target: str, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the exact background/holdout partition used by LP pretraining."""
    path = Path(state_root) / "_cache" / f"{target}_edge_split_s{seed}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"missing canonical LP edge split: {path}")
    split = torch.load(path, map_location="cpu", weights_only=False)
    train = split["train_edges"].long()
    holdout = split["validation_edges"].long()
    if train.ndim != 2 or holdout.ndim != 2 or train.shape[0] != 2 or holdout.shape[0] != 2:
        raise ValueError(f"malformed canonical LP edge split: {path}")
    return torch.cat((train, train.flip(0)), dim=1), holdout


def encoder(config: dict, checkpoint: dict, device: torch.device) -> GraphSAGE:
    protocol = dict(config["protocol"])
    protocol.update(checkpoint["metadata"].get("protocol", {}))
    model = GraphSAGE(
        int(protocol["input_dim"]), int(protocol["hidden_dim"]),
        int(protocol["output_dim"]), int(protocol["layers"]), float(protocol["dropout"]),
    )
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval()


@torch.no_grad()
def embed_full(model, graph, device) -> np.ndarray:
    return model(graph.x.float().to(device), graph.edge_index.long().to(device)).cpu().numpy()


@torch.no_grad()
def embed_nodes(model, graph, nodes: np.ndarray, protocol: dict, device) -> np.ndarray:
    loader = NeighborLoader(
        graph, input_nodes=torch.from_numpy(nodes),
        num_neighbors=[int(protocol["fanout"])] * int(protocol["layers"]),
        batch_size=int(protocol.get("eval_batch_size", 1024)), shuffle=False, num_workers=0,
    )
    table = np.empty((len(nodes), int(protocol["output_dim"])), dtype=np.float32)
    lookup = {int(node): index for index, node in enumerate(nodes.tolist())}
    for batch in loader:
        batch = batch.to(device)
        roots = batch.n_id[:batch.batch_size].cpu().numpy()
        encoded = model(batch.x.float(), batch.edge_index.long())[:batch.batch_size].cpu().numpy()
        table[[lookup[int(node)] for node in roots]] = encoded
    return table


def load_pair_module(prodigy_root: Path):
    path = prodigy_root / "scripts/eval/pair_link_eval.py"
    spec = importlib.util.spec_from_file_location("repaired_pair_link_eval", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves forward types through sys.modules during execution.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def eval_cls_target(target, args, config, rows, device, output_root):
    graph = unwrap(config["graphs"][target]["path"])
    y = graph.y.reshape(-1).cpu().numpy()
    labeled = np.isfinite(y) & (y >= 0)
    for run_id, _ in rows:
        output = output_root / "cls" / f"{run_id}__to__{target}.json"
        if output.is_file():
            continue
        checkpoint_path = Path(args.state_root) / args.objective / run_id / "best.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = encoder(config, checkpoint, device)
        embedding = embed_full(model, graph, device)[labeled]
        result = evaluate_embeddings(
            embedding, y[labeled], int(config["protocol"]["labels_per_class"]), args.seed
        )
        result.update({
            "status": "complete", "task": "classification", "target": target,
            "run_id": run_id, "objective": args.objective,
            "sources": checkpoint["metadata"]["sources"],
            "checkpoint_step": int(checkpoint["step"]), "seed": args.seed,
        })
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
        del model, embedding
        torch.cuda.empty_cache()


def eval_lp_target(target, args, config, rows, device, output_root, pair):
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
    background_graph = Data(x=graph.x.float(), edge_index=background_edges)
    baseline_path = output_root / "lp" / f"_protocol__{target}.json"
    if not baseline_path.is_file():
        baseline = {
            "target": target, "negative_kind": "degree_matched", "seed": args.seed,
            "edge_partition": "canonical_lp_training_cache",
            "n_pairs": len(pairs), "n_positive": int((pairs.label == 1).sum()),
            "holdout_leakage_edges": pair.leakage_check(background, pairs),
            "raw_feature_cosine": pair.evaluate_scores(
                "raw_feature_cosine", pairs.label,
                pair.pair_scores(graph.x.numpy(), pairs, "cosine"), val_mask,
            ).as_dict(),
        }
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps(baseline, indent=2) + "\n")
    protocol = config["protocol"]
    for run_id, _ in rows:
        output = output_root / "lp" / f"{run_id}__to__{target}.json"
        if output.is_file():
            continue
        checkpoint_path = Path(args.state_root) / args.objective / run_id / "best.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = encoder(config, checkpoint, device)
        table = embed_nodes(model, background_graph, nodes, protocol, device)
        embeddings = pair.NodeEmbeddings(table, nodes, n_nodes)
        scores = pair.pair_scores(embeddings, pairs, "cosine")
        report = pair.evaluate_scores("encoder_cosine", pairs.label, scores, val_mask).as_dict()
        result = {
            "status": "complete", "task": "static_link_prediction", "target": target,
            "run_id": run_id, "objective": args.objective,
            "sources": checkpoint["metadata"]["sources"],
            "checkpoint_step": int(checkpoint["step"]), "seed": args.seed,
            "edge_partition": "canonical_lp_training_cache",
            "negative_kind": "degree_matched", "n_pairs": len(pairs), "report": report,
            "gates": {
                "holdout_leakage_edges": pair.leakage_check(background, pairs),
                "endpoint_sensitivity": pair.endpoint_sensitivity(embeddings, pairs),
                "endpoint_permutation_auc": pair.endpoint_permutation_auc(
                    embeddings, pairs, np.random.default_rng(args.seed + 1)
                ),
            },
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
        del model, table, embeddings
        torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--objective", choices=("lp", "graphmae"), required=True)
    parser.add_argument("--task", choices=("cls", "lp"), required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--state-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--prodigy-root", default="/dataMeR1/phil/gfm/prodigy-nm-pairs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-id", help="evaluate only this lattice run (smoke/debug)")
    parser.add_argument("--target", help="evaluate only this target (smoke/debug)")
    args = parser.parse_args()
    config = load_config(args.config)
    configure_sampling_backend(config["protocol"])
    rows = lattice_rows()
    targets = CLS_TARGETS if args.task == "cls" else LP_TARGETS
    if args.run_id:
        rows = [row for row in rows if row[0] == args.run_id]
        if not rows:
            parser.error(f"unknown --run-id {args.run_id!r}")
    if args.target:
        if args.target not in targets:
            parser.error(f"invalid --target {args.target!r} for task {args.task!r}")
        targets = (args.target,)
    assigned = targets[args.worker_index::args.workers]
    pair = None if args.task == "cls" else load_pair_module(Path(args.prodigy_root))
    device = torch.device(f"cuda:{args.device}")
    for target in assigned:
        if args.task == "cls":
            eval_cls_target(target, args, config, rows, device, Path(args.output_root))
        else:
            eval_lp_target(target, args, config, rows, device, Path(args.output_root), pair)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
