#!/usr/bin/env python3
"""Extract support-independent VISION feature-encoder representations."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

from scripts.experiments.setup.icl_arch_matrix.architecture_adapters import build_adapter

from .protocol import FeatureCache, save_feature_cache
from .targets import graph_field, labeled_nodes, load_graph, load_labels, selected_targets


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, default=Path("/dataMeR1/phil/gfm/upstream/VISION"))
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--targets", default="covid_political,election2020,ukr_rus_suspended,twibot20")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8192)
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != "vision":
        raise ValueError(f"not a VISION checkpoint: {args.checkpoint}")
    adapter = build_adapter("vision", args.upstream_root)
    result = adapter.load_state_dict(checkpoint["model_state"], strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(result)
    encoder = adapter.model.feature_encoder.to(device).eval()
    checkpoint_hash = sha256_file(args.checkpoint)

    for target in selected_targets(args.targets):
        graph = load_graph(target)
        labels = load_labels(graph, target.label_key)
        nodes = labeled_nodes(labels)
        x = graph_field(graph, "x").float()
        mean = x.mean(dim=0, keepdim=True)
        outputs = []
        with torch.no_grad():
            for start in range(0, nodes.size, args.batch_size):
                ids = torch.from_numpy(nodes[start : start + args.batch_size])
                outputs.append(encoder((x[ids] - mean).to(device)).cpu())
        features = torch.cat(outputs).numpy()
        save_feature_cache(
            args.output_root / args.model_id / f"{target.name}.npz",
            FeatureCache(
                model_id=args.model_id,
                target=target.name,
                features=features,
                labels=labels,
                node_ids=nodes,
                metadata={
                    "architecture": "VISION",
                    "native_pretext": checkpoint.get("pretraining_protocol"),
                    "checkpoint": str(args.checkpoint),
                    "checkpoint_sha256": checkpoint_hash,
                    "checkpoint_step": int(checkpoint.get("step", -1)),
                    "training_seed": int(checkpoint.get("seed", -1)),
                    "representation": "learned_feature_encoder_after_full_graph_mean_centering",
                    "topology_used_by_probe_representation": False,
                    "label_key": target.label_key,
                },
            ),
        )
        print(f"{target.name}: labeled_nodes={nodes.size} dim={features.shape[1]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

