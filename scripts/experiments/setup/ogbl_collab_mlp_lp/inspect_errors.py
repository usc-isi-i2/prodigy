#!/usr/bin/env python3
"""Export inspectable mistakes from frozen ogbl-collab MLP checkpoints."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from scripts.experiments.setup.ogbl_collab_mlp_lp.run import (
    Encoder,
    audit_dataset,
    load_official_dataset,
    pair_keys,
    raw_cosine,
    score_edges,
    test_strata,
)


EXPECTED_FINGERPRINT = "07f7af8e654bda27caad60ed74c479f48780343826543dd613d90cb4e979f9f4"
ARM = "nonlinear_mlp_cosine"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hits_threshold(negative_scores: np.ndarray, k: int = 50) -> float:
    if not 0 < k <= len(negative_scores):
        raise ValueError(f"k={k} is invalid for {len(negative_scores)} negatives")
    return float(np.partition(negative_scores, len(negative_scores) - k)[-k])


def load_author_ids(path: Path, num_nodes: int) -> np.ndarray:
    values = np.empty(num_nodes, dtype=np.int64)
    seen = np.zeros(num_nodes, dtype=bool)
    with gzip.open(path, "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            index = int(row["node idx"])
            values[index] = int(row["author id"])
            seen[index] = True
    if not seen.all():
        raise ValueError(f"author mapping is missing {int((~seen).sum())} nodes")
    return values


def pair_event_counts(edges: np.ndarray, num_nodes: int) -> Counter[int]:
    return Counter(map(int, pair_keys(edges, num_nodes)))


def endpoint_event_degrees(edges: np.ndarray, num_nodes: int) -> np.ndarray:
    return np.bincount(np.asarray(edges).reshape(-1), minlength=num_nodes)


def select_examples(
    rows: list[dict], count: int, kind: str, stratum: str | None = None
) -> list[dict]:
    selected = [row for row in rows if row["kind"] == kind]
    if stratum is not None:
        selected = [row for row in selected if row["stratum"] == stratum]
    reverse = kind == "official_negative_false_positive"
    return sorted(selected, key=lambda row: row["mean_margin"], reverse=reverse)[:count]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("/dataMeR1/phil/data/ogb"))
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=262_144)
    parser.add_argument("--examples-per-group", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"choose a new output directory: {args.out}")
    if args.examples_per_group < 1:
        raise ValueError("examples-per-group must be positive")

    graph, split, _, ogb_version = load_official_dataset(args.dataset_root)
    audit = audit_dataset(graph, split, ogb_version)
    if audit["split_fingerprint"] != EXPECTED_FINGERPRINT:
        raise ValueError("dataset fingerprint differs from the frozen campaign")

    checkpoints = []
    for seed in range(3):
        seed_dir = args.campaign_root / f"seed{seed}"
        selection = json.loads((seed_dir / "selection_frozen.json").read_text())
        selected = next(row for row in selection["learned"] if row["arm"] == ARM)
        checkpoint = seed_dir / f"{ARM}.pt"
        if sha256_file(checkpoint) != selected["checkpoint_sha256"]:
            raise ValueError(f"checkpoint hash mismatch for seed {seed}")
        checkpoints.append((seed, checkpoint, selected["best_epoch"], selected["checkpoint_sha256"]))

    plan = {
        "dataset_fingerprint": audit["split_fingerprint"],
        "arm": ARM,
        "seeds": [seed for seed, *_ in checkpoints],
        "checkpoint_epochs": [epoch for _, _, epoch, _ in checkpoints],
        "test_positives": len(split["test"]["edge"]),
        "official_test_negatives": len(split["test"]["edge_neg"]),
        "device": args.device,
        "out": str(args.out),
    }
    if args.dry_run:
        print(json.dumps({"event": "dry_run", **plan}, indent=2, sort_keys=True))
        return

    device = torch.device(args.device)
    x = torch.as_tensor(np.asarray(graph["node_feat"]), dtype=torch.float32, device=device)
    positive_edges = np.asarray(split["test"]["edge"])
    negative_edges = np.asarray(split["test"]["edge_neg"])
    positive_scores, negative_scores, thresholds = [], [], []
    for seed, checkpoint, _, _ in checkpoints:
        payload = torch.load(checkpoint, map_location=device, weights_only=True)
        if payload["arm"] != ARM or payload["seed"] != seed:
            raise ValueError(f"checkpoint identity mismatch for seed {seed}")
        model = Encoder(ARM, x.shape[1]).to(device)
        model.load_state_dict(payload["state_dict"])
        positive_scores.append(score_edges(x, positive_edges, args.batch_size, model))
        negative_scores.append(score_edges(x, negative_edges, args.batch_size, model))
        thresholds.append(hits_threshold(negative_scores[-1]))
    positive_scores = np.stack(positive_scores)
    negative_scores = np.stack(negative_scores)
    thresholds_array = np.asarray(thresholds)[:, None]
    positive_margins = positive_scores - thresholds_array
    negative_margins = negative_scores - thresholds_array

    num_nodes = int(graph["num_nodes"])
    mapping = load_author_ids(
        args.dataset_root / "ogbl_collab" / "mapping" / "nodeidx2authorid.csv.gz", num_nodes
    )
    strata = test_strata(split["train"]["edge"], split["valid"]["edge"], positive_edges, num_nodes)
    train_pairs = pair_event_counts(split["train"]["edge"], num_nodes)
    valid_pairs = pair_event_counts(split["valid"]["edge"], num_nodes)
    train_degree = endpoint_event_degrees(split["train"]["edge"], num_nodes)
    valid_degree = endpoint_event_degrees(split["valid"]["edge"], num_nodes)

    rows: list[dict] = []
    for kind, edges, scores, margins in (
        ("test_positive_false_negative", positive_edges, positive_scores, positive_margins),
        ("official_negative_false_positive", negative_edges, negative_scores, negative_margins),
    ):
        if kind == "test_positive_false_negative":
            indices = np.flatnonzero(np.all(margins <= 0, axis=0))
        else:
            indices = np.flatnonzero(np.any(margins >= 0, axis=0))
        raw = raw_cosine(x[torch.as_tensor(edges[indices, 0], device=device)],
                         x[torch.as_tensor(edges[indices, 1], device=device)]).cpu().numpy()
        for offset, index in enumerate(indices):
            u, v = map(int, edges[index])
            key = int(pair_keys(edges[index:index + 1], num_nodes)[0])
            row = {
                "kind": kind,
                "split_index": int(index),
                "node_u": u,
                "node_v": v,
                "mag_author_u": int(mapping[u]),
                "mag_author_v": int(mapping[v]),
                "stratum": (
                    "seen_in_train" if kind.startswith("test_positive") and strata["seen_in_train"][index]
                    else "novel_vs_train" if kind.startswith("test_positive")
                    else "official_negative"
                ),
                "train_pair_events": train_pairs[key],
                "valid_pair_events": valid_pairs[key],
                "train_endpoint_events_u": int(train_degree[u]),
                "train_endpoint_events_v": int(train_degree[v]),
                "valid_endpoint_events_u": int(valid_degree[u]),
                "valid_endpoint_events_v": int(valid_degree[v]),
                "raw_feature_cosine": float(raw[offset]),
                "mean_score": float(scores[:, index].mean()),
                "mean_margin": float(margins[:, index].mean()),
                "wrong_seed_count": int((margins[:, index] <= 0).sum()) if kind.startswith("test_positive") else int((margins[:, index] >= 0).sum()),
            }
            for seed in range(3):
                row[f"seed{seed}_score"] = float(scores[seed, index])
                row[f"seed{seed}_threshold"] = thresholds[seed]
                row[f"seed{seed}_margin"] = float(margins[seed, index])
            rows.append(row)

    examples = []
    for stratum in ("seen_in_train", "novel_vs_train"):
        examples.extend(select_examples(rows, args.examples_per_group, "test_positive_false_negative", stratum))
    examples.extend(select_examples(rows, args.examples_per_group, "official_negative_false_positive"))

    args.out.mkdir(parents=True)
    fieldnames = list(examples[0])
    with (args.out / "examples.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(examples)
    receipt = {
        "complete": True,
        **plan,
        "checkpoint_sha256": [digest for *_, digest in checkpoints],
        "hits_at_50_threshold_by_seed": thresholds,
        "definition": "A positive is missed when its score is not strictly above the seed's 50th-highest official-negative score.",
        "counts": {
            "consensus_false_negative": int(np.all(positive_margins <= 0, axis=0).sum()),
            "consensus_false_negative_seen_in_train": int(np.sum(np.all(positive_margins <= 0, axis=0) & strata["seen_in_train"])),
            "consensus_false_negative_novel_vs_train": int(np.sum(np.all(positive_margins <= 0, axis=0) & strata["novel_vs_train"])),
            "official_negatives_above_threshold_any_seed": int(np.any(negative_margins >= 0, axis=0).sum()),
            "exported_examples": len(examples),
        },
        "limitations": [
            "Official negatives are benchmark decoys, not verified never-collaborated author pairs.",
            "The dataset exposes Microsoft Academic Graph author IDs but not author names or paper titles.",
        ],
    }
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
