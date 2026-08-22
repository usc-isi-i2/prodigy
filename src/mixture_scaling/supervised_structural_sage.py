from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .config import load_config
from .finetune_strict import (
    forward_labeled,
    label_mapping,
    metrics_from_logits,
    save,
    seed_everything,
)
from .model import GraphSAGE
from .strict_data import induced_partition, load_raw, load_split, split_hash
from .target_structural_sage import augment


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
        raise ValueError("supervised structural GraphSAGE may use only GPUs 2 and 3")
    run_dir = Path(args.output_dir)
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite run: {run_dir}")
    (run_dir / "checkpoints").mkdir(parents=True)
    config = load_config(args.config)
    protocol = config["protocol"]
    raw = load_raw(config["graphs"][args.target]["path"])
    split = load_split(Path(args.split_root) / f"{args.target}.pt", args.target, int(raw["x"].shape[0]))
    train_graph = induced_partition(args.target, config["graphs"][args.target]["path"], split, "train")
    validation_graph = induced_partition(
        args.target, config["graphs"][args.target]["path"], split, "validation"
    )
    feature_names = augment(train_graph)
    augment(validation_graph)
    classes, mapping = label_mapping(raw["y"].reshape(-1).long())
    input_dim = int(train_graph.data.x.shape[1])
    seed_everything(args.seed)
    device = torch.device(f"cuda:{args.device}")
    encoder = GraphSAGE(
        input_dim, int(protocol["hidden_dim"]), int(protocol["output_dim"]),
        int(protocol["layers"]), float(protocol["dropout"]),
    ).to(device)
    head = nn.Linear(int(protocol["output_dim"]), len(classes)).to(device)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(head.parameters()),
        lr=float(protocol["finetune_learning_rate"]),
        weight_decay=float(protocol["finetune_weight_decay"]),
    )
    metadata = {
        "run_id": "twibot_supervised_structural_plus_existing_s0", "target": args.target,
        "target_split_hash": split_hash(split), "seed": args.seed,
        "feature_mode": "structural_plus_existing", "input_dim": input_dim,
        "structural_feature_names": feature_names, "selection_metric": "validation_roc_auc_ovr_macro",
        "classes": classes, "test_evaluations": 0,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    best_auc, best_epoch, patience = float("-inf"), 0, 0
    start = time.monotonic()
    for epoch in range(1, int(protocol["finetune_max_epochs"]) + 1):
        encoder.train(); head.train(); optimizer.zero_grad(set_to_none=True)
        train_logits, train_y = forward_labeled(encoder, head, train_graph, mapping, device)
        train_loss = F.cross_entropy(train_logits, train_y)
        train_loss.backward(); optimizer.step()
        encoder.eval(); head.eval()
        with torch.no_grad():
            validation_logits, validation_y = forward_labeled(encoder, head, validation_graph, mapping, device)
            validation_loss = F.cross_entropy(validation_logits, validation_y)
            validation_metrics = metrics_from_logits(validation_logits, validation_y, classes)
        row = {
            "epoch": epoch, "train_loss": float(train_loss.detach()),
            "validation_loss": float(validation_loss.detach()),
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
            "elapsed_seconds": time.monotonic() - start,
        }
        with (run_dir / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        if epoch % int(protocol["finetune_checkpoint_interval"]) == 0:
            save(run_dir / "checkpoints" / f"epoch_{epoch}.pt", encoder, head, optimizer, epoch, metadata)
        auc = float(validation_metrics["roc_auc_ovr_macro"])
        if auc > best_auc + 1e-5:
            best_auc, best_epoch, patience = auc, epoch, 0
            save(run_dir / "best.pt", encoder, head, optimizer, epoch, metadata)
        else:
            patience += 1
        if epoch % 10 == 0:
            print(json.dumps(row), flush=True)
        if epoch >= int(protocol["finetune_min_epochs"]) and patience >= int(protocol["finetune_patience"]):
            break
    save(run_dir / "checkpoints" / f"epoch_{epoch}.pt", encoder, head, optimizer, epoch, metadata)
    selected = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    encoder.load_state_dict(selected["encoder"]); head.load_state_dict(selected["head"])

    # Construct and score test only after validation-AUC model selection.
    test_graph = induced_partition(args.target, config["graphs"][args.target]["path"], split, "test")
    augment(test_graph)
    encoder.eval(); head.eval()
    with torch.no_grad():
        test_logits, test_y = forward_labeled(encoder, head, test_graph, mapping, device)
        test_metrics = metrics_from_logits(test_logits, test_y, classes)
    summary = {
        **metadata, "status": "complete", "best_epoch": best_epoch,
        "best_validation_auc": best_auc, "final_epoch": epoch,
        "elapsed_seconds": time.monotonic() - start,
        "train_nodes": int((train_graph.data.y >= 0).sum()),
        "validation_nodes": int((validation_graph.data.y >= 0).sum()),
        "test_nodes": int((test_graph.data.y >= 0).sum()), "test_evaluations": 1,
        **test_metrics,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
