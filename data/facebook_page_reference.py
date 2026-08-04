"""Dataset adapter for the benchmark-ready Facebook page-reference graph."""

from __future__ import annotations

from data.ukr_rus_twitter import (
    get_ukr_rus_twitter_dataloader,
    get_ukr_rus_twitter_dataset,
)


def get_facebook_page_reference_dataset(root: str, **kwargs):
    kwargs.setdefault("graph_filename", "page_reference_graph.pt")
    return get_ukr_rus_twitter_dataset(root=root, **kwargs)


def get_facebook_page_reference_dataloader(*args, **kwargs):
    return get_ukr_rus_twitter_dataloader(*args, **kwargs)


__all__ = [
    "get_facebook_page_reference_dataset",
    "get_facebook_page_reference_dataloader",
]
