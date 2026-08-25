#!/usr/bin/env python3
"""Extract a reconstructed fixed-budget GraphSAGE pilot-v1 checkpoint trajectory."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from scripts.eval.pair_link_ckpt import load_graph_blob

from .extract_graphsage import node_batch, sha256_file, static_graph_arrays
from .protocol import FeatureCache, save_feature_cache
from .targets import labeled_nodes, load_labels, selected_targets


def parse_checkpoint(value: str) -> tuple[int, Path]:
    step_text, separator, path_text = value.partition("=")
    if not separator or not step_text.isdigit() or not path_text:
        raise argparse.ArgumentTypeError("checkpoint must be STEP=/absolute/path/checkpoint.pt")
    return int(step_text), Path(path_text)


def state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode("utf-8"))
        digest.update(state[name].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", type=parse_checkpoint, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--targets", default="covid_political,election2020,ukr_rus_suspended,twibot20"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()

    checkpoints = dict(args.checkpoint)
    expected_steps = {0, 20, 60, 100, 300, 900, 2000}
    if set(checkpoints) != expected_steps:
        raise ValueError(f"trajectory steps {sorted(checkpoints)} != {sorted(expected_steps)}")

    sys.path.insert(0, str(args.repository.resolve()))
    from socialgfm.benchmark_models import LinkPredictor  # noqa: PLC0415

    loaded = {}
    for step, path in sorted(checkpoints.items()):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("model") != "graphsage":
            raise ValueError(f"not a pilot GraphSAGE checkpoint: {path}")
        config = checkpoint["config"]
        if int(config["train_steps"]) != step:
            raise ValueError(f"checkpoint {path} declares {config['train_steps']} steps, expected {step}")
        loaded[step] = {
            "path": path,
            "checkpoint": checkpoint,
            "file_sha256": sha256_file(path),
            "state_sha256": state_sha256(checkpoint["model_state"]),
        }

    device = torch.device(args.device)
    for target in selected_targets(args.targets):
        blob, graph = load_graph_blob(str(target.graph))
        labels = load_labels(blob, target.label_key)
        nodes = labeled_nodes(labels)
        edge_index = torch.as_tensor(graph.edge_index).long()
        reference_config = loaded[2000]["checkpoint"]["config"]
        arrays = static_graph_arrays(
            edge_index, int(graph.num_nodes), int(reference_config["history_length"])
        )
        structural_graph = SimpleNamespace(
            record={"platform": "twitter", "key": target.name}, arrays=arrays
        )

        for step, record in sorted(loaded.items()):
            checkpoint = record["checkpoint"]
            config = checkpoint["config"]
            model = LinkPredictor(
                int(config["hidden_dim"]), int(config["history_length"]), False, False
            ).to(device)
            model.load_state_dict(checkpoint["model_state"], strict=True)
            model.eval()
            outputs = []
            with torch.no_grad():
                for start in range(0, nodes.size, args.batch_size):
                    ids = nodes[start : start + args.batch_size]
                    outputs.append(
                        model.encoder(node_batch(structural_graph, ids, device)).cpu()
                    )
            features = torch.cat(outputs).numpy()
            model_id = f"graphsage_pilot_v1_step{step}"
            save_feature_cache(
                args.output_root / model_id / f"{target.name}.npz",
                FeatureCache(
                    model_id=model_id,
                    target=target.name,
                    features=features,
                    labels=labels,
                    node_ids=nodes,
                    metadata={
                        "architecture": "GraphSAGE",
                        "native_pretext": "pilot_v1_link_prediction",
                        "checkpoint": str(record["path"]),
                        "checkpoint_sha256": record["file_sha256"],
                        "state_sha256": record["state_sha256"],
                        "training_seed": int(config["seed"]),
                        "training_steps": step,
                        "trajectory_kind": "exact_deterministic_prefix_reconstruction",
                        "reconstruction_source_commit": "c6fd912fba1c12b12b8a6e9b0d112b63b0c563a3",
                        "terminal_matches_registered_checkpoint": step == 2000,
                        "representation": "frozen_node_history_encoder",
                        "edge_view": "graph.edge_index",
                        "relation_mapping": "all target edges -> reshare",
                        "node_type_mapping": "all target nodes -> actor",
                        "history_length": int(config["history_length"]),
                        "label_key": target.label_key,
                    },
                ),
            )
            print(
                f"{model_id}/{target.name}: labeled_nodes={nodes.size} dim={features.shape[1]}",
                flush=True,
            )
            del model, outputs, features
            if device.type == "cuda":
                torch.cuda.empty_cache()

        del blob, graph, arrays, structural_graph
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
