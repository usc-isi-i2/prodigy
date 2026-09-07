"""Canonical downstream label targets shared by every adaptation arm."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class Target:
    name: str
    graph: Path
    label_key: str = "y"


TARGETS = {
    target.name: target
    for target in (
        Target("covid_political", Path("/dataMeR1/phil/data/covid_political/graphs/retweet_graph.pt")),
        Target("election2020", Path("/dataMeR1/phil/data/election2020/graphs/retweet_graph.pt")),
        Target("ukr_rus_suspended", Path("/dataMeR1/phil/data/ukr_rus_suspended/graphs/retweet_graph.pt")),
        Target("twibot20", Path("/dataMeR1/phil/data/twibot20/graphs/retweet_graph.pt")),
        Target(
            "facebook_page_category_top30",
            Path("/dataMeR1/phil/data/facebook_page_reference/graphs/page_reference_graph.pt"),
            "node_classification_targets.page_category_top30",
        ),
        Target(
            "facebook_admin_country_top30",
            Path("/dataMeR1/phil/data/facebook_page_reference/graphs/page_reference_graph.pt"),
            "node_classification_targets.admin_country_top30",
        ),
        Target(
            "facebook_verified",
            Path("/dataMeR1/phil/data/facebook_page_reference/graphs/page_reference_graph.pt"),
            "node_classification_targets.verified",
        ),
        Target("cora", Path("/dataMeR1/phil/data/cora/graphs/citation_graph.pt")),
        Target("pubmed", Path("/dataMeR1/phil/data/pubmed/graphs/citation_graph.pt")),
    )
}


def nested_value(value: Any, dotted_key: str) -> Any:
    current = value
    for part in dotted_key.split("."):
        if isinstance(current, dict):
            current = current[part]
        else:
            current = getattr(current, part)
    return current


def load_graph(target: Target):
    return torch.load(target.graph, map_location="cpu", weights_only=False)


def graph_field(graph: Any, name: str):
    if isinstance(graph, dict):
        if name in graph:
            return graph[name]
        for key in ("data", "graph"):
            nested = graph.get(key)
            if nested is not None:
                if isinstance(nested, dict) and name in nested:
                    return nested[name]
                value = getattr(nested, name, None)
                if value is not None:
                    return value
    value = getattr(graph, name, None)
    if value is None:
        raise KeyError(f"graph has no {name!r}")
    return value


def load_labels(graph: Any, label_key: str) -> np.ndarray:
    labels = nested_value(graph, label_key)
    if not isinstance(labels, torch.Tensor):
        labels = torch.as_tensor(labels)
    return labels.detach().cpu().reshape(-1).to(torch.long).numpy()


def labeled_nodes(labels: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.asarray(labels) >= 0).astype(np.int64)


def selected_targets(text: str) -> list[Target]:
    names = [name for name in text.split(",") if name] if text else list(TARGETS)
    unknown = sorted(set(names) - set(TARGETS))
    if unknown:
        raise ValueError(f"unknown targets: {unknown}")
    return [TARGETS[name] for name in names]

