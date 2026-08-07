"""Shared PRODIGY loader for GTE-attributed Cora and PubMed graphs."""

from __future__ import annotations

from .social_llm_dataset import _get_dataloader, _get_dataset


def _get_citation_dataset(dataset_name: str, root: str, n_hop: int,
                          graph_filename: str, **kwargs):
    # The global CLI always supplies edge_view="default".  Static LP must use
    # the leakage-free background even in that case; the held-out view remains
    # the positive target sampler.
    if (
        kwargs.get("task_name") == "static_link_prediction"
        and kwargs.get("edge_view", "default") in {None, "", "default"}
    ):
        kwargs["edge_view"] = "static_background"
    return _get_dataset(
        dataset_name,
        root,
        n_hop=n_hop,
        graph_filename=graph_filename,
        **kwargs,
    )


def get_cora_dataset(root: str, n_hop: int = 1,
                     graph_filename: str = "citation_graph.pt", **kwargs):
    return _get_citation_dataset(
        "cora", root, n_hop, graph_filename, **kwargs
    )


def get_cora_dataloader(*args, **kwargs):
    return _get_dataloader("cora", *args, **kwargs)


def get_pubmed_dataset(root: str, n_hop: int = 1,
                       graph_filename: str = "citation_graph.pt", **kwargs):
    return _get_citation_dataset(
        "pubmed", root, n_hop, graph_filename, **kwargs
    )


def get_pubmed_dataloader(*args, **kwargs):
    return _get_dataloader("pubmed", *args, **kwargs)


__all__ = [
    "get_cora_dataset",
    "get_cora_dataloader",
    "get_pubmed_dataset",
    "get_pubmed_dataloader",
]
