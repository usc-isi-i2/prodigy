#!/usr/bin/env python3
"""Evaluate directed-degree-only baselines on the fixed CLS episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from data.structural_features import compute_structural_features
from scripts.experiments.setup.icl_arch_matrix.common_protocol import (
    EVAL_N_SHOT,
    EVAL_N_WAY,
    build_classification_dataset,
    build_classification_loader,
    classification_targets,
    iter_episodes,
    new_fingerprint,
    reset_episode_rng,
    update_episode_fingerprint,
)


BASELINES = ("topology_degree_prototype", "topology_degree_logistic")
FEATURE_VIEW = "full_graph_directed_log_degree3_zscore"
FEATURE_NAMES = ("log1p_in_degree", "log1p_out_degree", "log1p_total_degree")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/dataMeR1/phil/data")
    parser.add_argument("--catalog", default="docs/graph_catalog.json")
    parser.add_argument("--results", required=True)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--datasets", default="")
    return parser.parse_args()


def _topology_features(dataset) -> torch.Tensor:
    graph = dataset.graph
    features = compute_structural_features(
        graph.edge_index.cpu(),
        int(graph.num_nodes),
        mode="directed3_log",
        standardize=True,
    )
    if features.shape != (int(graph.num_nodes), len(FEATURE_NAMES)):
        raise ValueError(f"unexpected topology feature shape: {tuple(features.shape)}")
    if not torch.isfinite(features).all():
        raise ValueError("non-finite topology features")
    return features


def _prototype_logits(
    support_x: torch.Tensor,
    support_y: torch.Tensor,
    query_x: torch.Tensor,
) -> torch.Tensor:
    support_x = F.normalize(support_x, p=2, dim=1)
    query_x = F.normalize(query_x, p=2, dim=1)
    prototypes = torch.stack(
        [support_x[support_y == label].mean(0) for label in range(EVAL_N_WAY)]
    )
    prototypes = F.normalize(prototypes, p=2, dim=1)
    return query_x @ prototypes.t()


def _append_metrics(store, baseline, target_global, probability_global1, prediction_global):
    store[baseline]["labels"].extend(target_global.tolist())
    store[baseline]["scores"].extend(probability_global1.tolist())
    store[baseline]["predictions"].extend(prediction_global.tolist())


def evaluate(loader, topology_features: torch.Tensor, *, n_query: int, equal_query_counts: bool):
    store = {
        baseline: {"labels": [], "scores": [], "predictions": []}
        for baseline in BASELINES
    }
    fingerprint = new_fingerprint()
    episode_count = 0
    reset_episode_rng()
    for batch in loader:
        for episode in iter_episodes(
            batch,
            n_way=EVAL_N_WAY,
            n_shot=EVAL_N_SHOT,
            n_query=n_query,
            equal_query_counts=equal_query_counts,
        ):
            center_x = topology_features[episode.global_centers.long()].float()
            support_x = center_x[episode.support_mask]
            query_x = center_x[episode.query_mask]
            support_y = episode.labels[episode.support_mask]
            query_y = episode.labels[episode.query_mask]
            target_global = episode.label_map[query_y].cpu().numpy().astype(np.int64)
            global_class1_local = torch.where(episode.label_map == 1)[0]
            if global_class1_local.numel() != 1:
                raise ValueError(f"expected one global class 1: {episode.label_map.tolist()}")
            positive_local = int(global_class1_local.item())

            logits = _prototype_logits(support_x, support_y, query_x)
            probabilities = torch.softmax(logits, dim=1)
            prototype_prediction = episode.label_map[logits.argmax(1)].cpu().numpy()
            _append_metrics(
                store,
                "topology_degree_prototype",
                target_global,
                probabilities[:, positive_local].cpu().numpy(),
                prototype_prediction,
            )

            logistic = LogisticRegression(
                C=1.0,
                penalty="l2",
                solver="liblinear",
                fit_intercept=True,
                max_iter=1000,
                tol=1e-6,
                random_state=0,
            )
            logistic.fit(support_x.cpu().numpy(), support_y.cpu().numpy())
            logistic_probability = logistic.predict_proba(query_x.cpu().numpy())
            class_columns = {int(label): index for index, label in enumerate(logistic.classes_)}
            logistic_prediction_local = logistic.predict(query_x.cpu().numpy()).astype(np.int64)
            logistic_prediction_global = (
                episode.label_map[torch.from_numpy(logistic_prediction_local)].cpu().numpy()
            )
            _append_metrics(
                store,
                "topology_degree_logistic",
                target_global,
                logistic_probability[:, class_columns[positive_local]],
                logistic_prediction_global,
            )

            update_episode_fingerprint(fingerprint, episode)
            episode_count += 1

    metrics = {}
    for baseline, values in store.items():
        y_true = np.asarray(values["labels"], dtype=np.int64)
        y_score = np.asarray(values["scores"], dtype=np.float64)
        y_pred = np.asarray(values["predictions"], dtype=np.int64)
        metrics[baseline] = {
            "episodes": episode_count,
            "queries": int(y_true.size),
            "episode_fingerprint": fingerprint.hexdigest(),
            "roc_auc": float(roc_auc_score(y_true, y_score)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        }
    return metrics


def main() -> int:
    args = parse_args()
    torch.set_num_threads(16)
    result_path = Path(args.results)
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite results: {result_path}")
    result_path.parent.mkdir(parents=True, exist_ok=True)

    targets = classification_targets(args.catalog)
    selected = set(filter(None, args.datasets.split(",")))
    if selected:
        missing = selected - targets.keys()
        if missing:
            raise ValueError(f"unknown classification datasets: {sorted(missing)}")
        targets = {name: target for name, target in targets.items() if name in selected}

    with result_path.open("w", encoding="utf-8") as handle:
        for dataset_name, target in targets.items():
            dataset, get_dataloader, graph_path = build_classification_dataset(
                dataset_name=dataset_name,
                data_root=args.data_root,
                target=target,
            )
            topology_features = _topology_features(dataset)
            loader = build_classification_loader(
                dataset_name=dataset_name,
                data_root=args.data_root,
                target=target,
                dataset=dataset,
                get_dataloader=get_dataloader,
                graph_path=graph_path,
                workers=args.workers,
            )
            results = evaluate(
                loader,
                topology_features,
                n_query=int(target["n_query"]),
                equal_query_counts=not target["eval_random_query"],
            )
            for baseline in BASELINES:
                row = {
                    "baseline": baseline,
                    "model_id": baseline,
                    "sources": [],
                    "seed": 0,
                    "training_updates": 0,
                    "feature_view": FEATURE_VIEW,
                    "feature_names": list(FEATURE_NAMES),
                    "topology_scope": "entire_loaded_target_graph",
                    "topology_used": True,
                    "raw_features_used": False,
                    "support_fit": "none" if baseline.endswith("prototype") else "logistic_c1",
                    "task": "classification",
                    "dataset": dataset_name,
                    "n_way": EVAL_N_WAY,
                    "n_shot": EVAL_N_SHOT,
                    "n_query": int(target["n_query"]),
                    **results[baseline],
                }
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
                print(json.dumps(row, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
