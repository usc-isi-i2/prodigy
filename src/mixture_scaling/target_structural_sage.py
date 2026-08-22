from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .config import load_config
from .model import GraphSAGE
from .probe_strict import choose_trial, classifier, embed
from .strict_data import induced_partition, load_raw, load_split, split_hash
from .strict_metrics import classification_metrics
from .structural_logistic_baseline import structural_features
from .train import batch_loss, configure_sampling_backend, make_loader, next_batch, save_checkpoint, validation_loss
from .train_strict import seed_everything, update_selection


def augment(
    graph,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    structure, names = structural_features(graph.data.edge_index, int(graph.data.num_nodes))
    if mean is None or std is None:
        mean = structure.mean(axis=0, dtype=np.float64)
        std = structure.std(axis=0, dtype=np.float64)
        std[std < 1e-8] = 1.0
    normalized = ((structure - mean) / std).astype(np.float32)
    graph.data.x = torch.cat((graph.data.x, torch.from_numpy(normalized)), dim=1)
    return names, mean, std


def select_probe(model, train_graph, validation_graph, device, c_values, seed):
    train_x, train_y = embed(model, train_graph, device)
    validation_x, validation_y = embed(model, validation_graph, device)
    trials = []
    for c_value in map(float, c_values):
        head = classifier(c_value, seed)
        head.fit(train_x, train_y)
        trials.append({
            "c": c_value,
            **classification_metrics(
                validation_y, head.predict(validation_x), head.predict_proba(validation_x), head.classes_
            ),
        })
    return choose_trial(trials, "auc"), trials, train_x, train_y, validation_x, validation_y


def score_probe(selection, model, test_graph, device, seed):
    chosen, trials, train_x, train_y, validation_x, validation_y = selection
    head = classifier(float(chosen["c"]), seed)
    head.fit(np.concatenate((train_x, validation_x)), np.concatenate((train_y, validation_y)))
    test_x, test_y = embed(model, test_graph, device)
    return {
        "selected_c": chosen["c"],
        "selection_metric": "validation_roc_auc_ovr_macro",
        "validation_trials": trials,
        "train_nodes": int(len(train_y)), "validation_nodes": int(len(validation_y)),
        "test_nodes": int(len(test_y)),
        **classification_metrics(test_y, head.predict(test_x), head.predict_proba(test_x), head.classes_),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split-root", required=True)
    parser.add_argument("--target", default="twibot20")
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.device not in (2, 3):
        raise ValueError("structural GraphSAGE pilot may use only GPUs 2 and 3")
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite run: {output_dir}")
    (output_dir / "checkpoints").mkdir(parents=True)
    config = load_config(args.config)
    protocol = dict(config["protocol"])
    configure_sampling_backend(protocol)
    raw = load_raw(config["graphs"][args.target]["path"])
    split = load_split(Path(args.split_root) / f"{args.target}.pt", args.target, int(raw["x"].shape[0]))
    train_graph = induced_partition(args.target, config["graphs"][args.target]["path"], split, "train")
    validation_graph = induced_partition(
        args.target, config["graphs"][args.target]["path"], split, "validation"
    )
    feature_names, structural_mean, structural_std = augment(train_graph)
    augment(validation_graph, structural_mean, structural_std)
    input_dim = int(train_graph.data.x.shape[1])
    device = torch.device(f"cuda:{args.device}")

    seed_everything(args.seed)
    model = GraphSAGE(
        input_dim, int(protocol["hidden_dim"]), int(protocol["output_dim"]),
        int(protocol["layers"]), float(protocol["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(protocol["learning_rate"]), weight_decay=float(protocol["weight_decay"])
    )
    train_loader = make_loader(train_graph, protocol, validation=False)
    validation_loader = make_loader(validation_graph, protocol, validation=True)
    metadata = {
        "run_id": f"{args.target}_target_structural_plus_existing_s{args.seed}", "sources": [args.target],
        "seed": args.seed, "protocol": protocol, "split_hashes": {args.target: split_hash(split)},
        "ssl_train_partition": "train", "ssl_validation_partition": "validation",
        "source_confined": True, "feature_mode": "structural_plus_existing",
        "input_dim": input_dim, "structural_feature_names": feature_names,
        "structural_normalization": "train_partition_zscore",
        "structural_mean": structural_mean.tolist(), "structural_std": structural_std.tolist(),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    iterator = None
    best_loss, best_step, patience_reference, patience = float("inf"), 0, float("inf"), 0
    checkpoints = set(map(int, protocol["checkpoint_steps"]))
    start = time.monotonic()
    for step in range(1, int(protocol["max_steps"]) + 1):
        batch, iterator = next_batch(train_loader, iterator)
        optimizer.zero_grad(set_to_none=True)
        loss = batch_loss(model, batch, device)
        loss.backward()
        optimizer.step()
        if step in checkpoints:
            save_checkpoint(output_dir / "checkpoints" / f"step_{step}.pt", model, optimizer, step, metadata)
        if step % int(protocol["validation_interval"]):
            continue
        value, per_source = validation_loss(
            model, [validation_loader], device, int(protocol["validation_batches"]), args.seed
        )
        row = {
            "step": step, "train_loss": float(loss), "validation_loss": value,
            "per_source_validation_loss": {args.target: next(iter(per_source.values()))},
            "elapsed_seconds": time.monotonic() - start,
        }
        with (output_dir / "validation.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        best_loss, best_step, patience_reference, patience, save_best = update_selection(
            value, step, best_loss, best_step, patience_reference, patience,
            float(protocol["min_relative_improvement"]),
        )
        if save_best:
            save_checkpoint(output_dir / "best.pt", model, optimizer, step, metadata)
        print(json.dumps(row), flush=True)
        if step >= int(protocol["minimum_steps"]) and patience >= int(protocol["patience_evaluations"]):
            break
    save_checkpoint(output_dir / "checkpoints" / f"step_{step}.pt", model, optimizer, step, metadata)
    summary = {
        **metadata, "status": "complete", "best_step": best_step,
        "best_validation_loss": best_loss, "final_step": step,
        "elapsed_seconds": time.monotonic() - start,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    selected = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=False)
    pretrained = GraphSAGE(
        input_dim, int(protocol["hidden_dim"]), int(protocol["output_dim"]),
        int(protocol["layers"]), float(protocol["dropout"]),
    )
    pretrained.load_state_dict(selected["model"])
    pretrained.to(device)
    seed_everything(args.seed)
    scratch = GraphSAGE(
        input_dim, int(protocol["hidden_dim"]), int(protocol["output_dim"]),
        int(protocol["layers"]), float(protocol["dropout"]),
    ).to(device)
    pretrained_selection = select_probe(
        pretrained, train_graph, validation_graph, device, protocol["probe_c_values"], args.seed
    )
    scratch_selection = select_probe(
        scratch, train_graph, validation_graph, device, protocol["probe_c_values"], args.seed
    )

    # Construct the test graph and read its labels only after both heads are selected.
    test_graph = induced_partition(args.target, config["graphs"][args.target]["path"], split, "test")
    augment(test_graph, structural_mean, structural_std)
    for name, encoder, selection, checkpoint_step in (
        ("pretrained", pretrained, pretrained_selection, best_step),
        ("scratch", scratch, scratch_selection, 0),
    ):
        result = {
            "status": "complete", "evaluation": "frozen_linear_probe",
            "target": args.target, "target_split_hash": split_hash(split),
            "feature_mode": "structural_plus_existing", "input_dim": input_dim,
            "encoder": name, "seed": args.seed, "checkpoint_step": checkpoint_step,
            **score_probe(selection, encoder, test_graph, device, args.seed),
        }
        (output_dir / f"probe_{name}.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
