#!/usr/bin/env python3
"""Evaluate a VISION/GILT specialist's native NM trajectory."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from scripts.experiments.setup.icl_arch_matrix.architecture_adapters import build_adapter
from scripts.experiments.setup.icl_arch_matrix.common_protocol import (
    EVAL_BATCH_SIZE,
    EVAL_EPISODES,
    EPISODE_RNG_SEED,
    N_QUERY,
    N_SHOT,
    N_WAY,
    build_dataset,
    build_loader,
    iter_episodes,
    load_config,
    new_fingerprint,
    reset_episode_rng,
    update_episode_fingerprint,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=("vision", "gilt"), required=True)
    parser.add_argument("--upstream-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--state-root", required=True, type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--training-seed", required=True, type=int)
    parser.add_argument("--checkpoint-steps", default="0,20,60,100,300,900,2000")
    parser.add_argument("--eval-episode-seed-offset", type=int, default=0)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    steps = sorted({int(value) for value in args.checkpoint_steps.split(",")})
    config = load_config(args.config)
    dataset = build_dataset(config)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    args.results.parent.mkdir(parents=True, exist_ok=True)
    if args.results.exists():
        raise FileExistsError(args.results)
    with args.results.open("w", encoding="utf-8") as handle:
        for step in steps:
            torch.manual_seed(args.training_seed)
            torch.cuda.manual_seed_all(args.training_seed)
            model = build_adapter(args.architecture, args.upstream_root)
            checkpoint = None
            if step:
                checkpoint = (
                    args.state_root / args.architecture / f"{args.model_id}_s{args.training_seed}"
                    / "checkpoint" / f"state_dict_{step}.pt"
                )
                state = torch.load(checkpoint, map_location="cpu")
                model.load_state_dict(state["model_state"], strict=True)
            model.to(device).eval()
            reset_episode_rng(EPISODE_RNG_SEED + args.eval_episode_seed_offset)
            loader = build_loader(
                dataset,
                config,
                split="test",
                sources=args.source,
                batch_count=EVAL_EPISODES // EVAL_BATCH_SIZE,
                batch_size=EVAL_BATCH_SIZE,
                workers=0,
                eval_episode_seed_offset=args.eval_episode_seed_offset,
            )
            labels, scores, predictions = [], [], []
            fingerprint = new_fingerprint()
            episodes = 0
            with torch.no_grad():
                for batch in loader:
                    graphs = batch[0].to(device)
                    moved = (graphs,) + tuple(
                        value.to(device) if torch.is_tensor(value) else value
                        for value in batch[1:]
                    )
                    for episode in iter_episodes(moved):
                        output = model.episode_logits(episode)
                        logits = output[0] if isinstance(output, tuple) else output
                        target = episode.labels[episode.query_mask]
                        probability = torch.softmax(logits, dim=1)
                        labels.extend(target.cpu().tolist())
                        scores.extend(probability.cpu().tolist())
                        predictions.extend(logits.argmax(1).cpu().tolist())
                        update_episode_fingerprint(fingerprint, episode)
                        episodes += 1
            y_true = np.asarray(labels, dtype=np.int64)
            y_score = np.asarray(scores, dtype=np.float64)
            y_pred = np.asarray(predictions, dtype=np.int64)
            row = {
                "architecture": args.architecture,
                "model_id": args.model_id,
                "sources": [args.source],
                "training_seed": args.training_seed,
                "eval_episode_seed_offset": args.eval_episode_seed_offset,
                "checkpoint_step": step,
                "checkpoint": str(checkpoint) if checkpoint else None,
                "baseline": "random_init" if step == 0 else "pretrained",
                "task": "neighbor_matching",
                "dataset": args.source,
                "n_way": N_WAY,
                "n_shot": N_SHOT,
                "n_query": N_QUERY,
                "episodes": episodes,
                "queries": int(y_true.size),
                "episode_fingerprint": fingerprint.hexdigest(),
                "roc_auc": float(roc_auc_score(y_true, y_score, multi_class="ovr", average="macro")),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            print(json.dumps(row, sort_keys=True), flush=True)
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
