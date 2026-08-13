"""Shared-batch scoring primitives for CPU-bound checkpoint evaluation."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np
import torch
from tqdm import tqdm


def to_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def score_models_on_shared_batches(
    *,
    models: dict[int, torch.nn.Module],
    steps: Sequence[int],
    dataloader: Iterable,
    device: torch.device,
    get_loss_and_score: Callable,
    get_aux_loss: Callable,
    compute_metrics: Callable | None = None,
) -> dict[int, dict[str, float]]:
    """Run every model on the same collated stream and aggregate eval metrics."""
    y_true: dict[int, list[torch.Tensor]] = {step: [] for step in steps}
    y_pred: dict[int, list[torch.Tensor]] = {step: [] for step in steps}
    batch_scores: dict[int, list[float]] = {step: [] for step in steps}
    aux_losses: dict[int, list[float]] = {step: [] for step in steps}

    with torch.no_grad():
        for batch in tqdm(dataloader, leave=False):
            batch = [value.to(device) for value in batch]
            graph = batch[0]
            original_x = graph.x
            original_edge_attr = getattr(graph, "edge_attr", None)
            for step in steps:
                # SingleLayerGeneralGNN writes hidden states back to graph.x.
                # Restore immutable collated inputs before every checkpoint.
                graph.x = original_x
                if hasattr(graph, "edge_attr"):
                    graph.edge_attr = original_edge_attr
                yt, yp, output_graph = models[step](*batch)
                _, batch_score = get_loss_and_score(yt, yp)
                y_true[step].append(yt.detach())
                y_pred[step].append(yp.detach())
                batch_scores[step].append(to_float(batch_score))
                aux_losses[step].append(to_float(get_aux_loss(output_graph)))
            graph.x = original_x
            if hasattr(graph, "edge_attr"):
                graph.edge_attr = original_edge_attr

    results = {}
    for step in steps:
        yt = torch.cat(y_true[step], dim=0)
        yp = torch.cat(y_pred[step], dim=0)
        loss, score = get_loss_and_score(yt, yp)
        results[step] = {
            "score": to_float(score),
            "score_std": float(np.std(batch_scores[step])),
            "loss": to_float(loss),
            "aux_loss": float(np.mean(aux_losses[step])),
        }
        if compute_metrics is not None:
            results[step].update(
                {
                    key: to_float(value)
                    for key, value in compute_metrics(yt, yp).items()
                }
            )
    return results
