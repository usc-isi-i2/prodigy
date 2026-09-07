#!/usr/bin/env python3
"""Extract frozen native-GraphCL SAMGPT base-GCN representations."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import yaml

from .protocol import FeatureCache, save_feature_cache
from .targets import labeled_nodes, load_labels, selected_targets


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--resolved-config", type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--targets", default="covid_political,election2020,ukr_rus_suspended,twibot20")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    sys.path.insert(0, str(args.repository.resolve()))
    from samgpt_social.data import (  # noqa: PLC0415
        gaussian_projection_matrix,
        load_graph_artifact,
        make_undirected_adjacency,
    )
    from samgpt_social.graphcl_ladder_eval import load_checkpoint_state  # noqa: PLC0415
    from samgpt_social.runner import (  # noqa: PLC0415
        build_preprompt_model,
        embed_target,
        prepare_features,
    )

    training = yaml.safe_load(args.training_config.read_text(encoding="utf-8"))
    resolved_path = args.resolved_config or args.checkpoint.with_name("resolved_config.json")
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    projection = gaussian_projection_matrix(
        768, int(training["feature_components"]), int(training["feature_projection_seed"])
    )
    model = build_preprompt_model(
        int(training["feature_components"]),
        int(training["pretrain_dataset_slots"]),
        resolved,
        device,
    )
    model.load_state_dict(load_checkpoint_state(args.checkpoint, device))
    checkpoint_hash = sha256_file(args.checkpoint)

    for target in selected_targets(args.targets):
        graph = load_graph_artifact(target.graph)
        labels = load_labels(graph, target.label_key)
        nodes = labeled_nodes(labels)
        features = prepare_features(graph, projection)
        adjacency = make_undirected_adjacency(graph.edge_index, graph.num_nodes)
        all_embeddings = embed_target(model, features, adjacency, device)
        embeddings = all_embeddings[torch.from_numpy(nodes)].numpy()
        save_feature_cache(
            args.output_root / args.model_id / f"{target.name}.npz",
            FeatureCache(
                model_id=args.model_id,
                target=target.name,
                features=embeddings,
                labels=labels,
                node_ids=nodes,
                metadata={
                    "architecture": "SAMGPT",
                    "native_pretext": "GraphCL",
                    "checkpoint": str(args.checkpoint),
                    "checkpoint_sha256": checkpoint_hash,
                    "checkpoint_update": int(training["optimizer_updates"]),
                    "training_seed": args.training_seed,
                    "representation": "frozen_base_gcn",
                    "graph_direction": "symmetrized",
                    "feature_projection": "shared_gaussian_random",
                    "feature_projection_seed": int(training["feature_projection_seed"]),
                    "label_key": target.label_key,
                },
            ),
        )
        print(f"{target.name}: labeled_nodes={nodes.size} dim={embeddings.shape[1]}", flush=True)
        del graph, features, adjacency, all_embeddings
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

