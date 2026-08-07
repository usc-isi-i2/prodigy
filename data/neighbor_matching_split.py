"""Leakage-free edge views for neighbor-matching episodes.

Historically the NM dataloaders used ``split`` only to seed a different episode
stream; train/val/test all sampled positives from the same adjacency.  This
module provides an explicit opt-in protocol:

* message passing and training positives use ``static_background``;
* validation/test positives use ``static_holdout``; and
* the two samplers share nodes/features but never the held-out edges.

Dataset loaders call :func:`configure_edge_split` while the raw artifact is in
memory.  Dataloaders call :func:`positive_sampler_for_split` when constructing
the NM task.  The default is disabled, preserving all historical behavior.
"""

from __future__ import annotations

from typing import Any

from torch_geometric.data import Data

from experiments.sampler import NeighborSampler


BACKGROUND_VIEW = "static_background"
HOLDOUT_VIEW = "static_holdout"
TRAIN_VIEW = "static_train"
VALIDATION_VIEW = "static_validation"
TEST_VIEW = "static_test"


def enabled(kwargs: dict[str, Any]) -> bool:
    return bool(kwargs.get("neighbor_matching_edge_split", False))


def _target_edge_index(raw: Any, view: str):
    if isinstance(raw, dict):
        views = raw.get("target_edge_index_views", {})
        if isinstance(views, dict) and view in views:
            return views[view]
        legacy = raw.get(f"edge_index_{view}")
        if legacy is not None:
            return legacy
    views = getattr(raw, "target_edge_index_views", {})
    if isinstance(views, dict) and view in views:
        return views[view]
    return getattr(raw, f"edge_index_{view}", None)


def ensure_static_views(raw: Any, kwargs: dict[str, Any]) -> None:
    """Derive the canonical static split in memory when an artifact lacks views.

    Some smaller source artifacts predate the benchmark-view enrichment.  Split-aware
    NM must still be usable without rewriting those shared files.  Existing views are
    authoritative; only a completely missing pair is derived.  A partial pair fails.
    """
    if not enabled(kwargs):
        return
    if not isinstance(raw, dict):
        raise TypeError("split-aware neighbor matching requires a dict graph artifact")
    if str(kwargs.get("edge_view", "default")) == TRAIN_VIEW:
        train = raw.get("edge_index_views", {}).get(TRAIN_VIEW)
        validation = raw.get("target_edge_index_views", {}).get(VALIDATION_VIEW)
        test = raw.get("target_edge_index_views", {}).get(TEST_VIEW)
        if train is None or validation is None or test is None:
            raise ValueError(
                "Three-way neighbor matching requires stored static_train, "
                "static_validation, and static_test views; refusing an in-memory "
                "split of a production artifact."
            )
        return
    background = raw.get("edge_index_views", {}).get(BACKGROUND_VIEW)
    holdout = raw.get("target_edge_index_views", {}).get(HOLDOUT_VIEW)
    if (background is None) != (holdout is None):
        raise ValueError(
            "Graph artifact contains only one of static_background/static_holdout; "
            "refusing to guess how the partial split was constructed."
        )
    if background is not None:
        return
    edge_index = raw.get("edge_index")
    if edge_index is None:
        raise ValueError("cannot derive a static split: graph artifact has no edge_index")

    from scripts.graph_construction.benchmark_targets import build_static_edge_split

    split = build_static_edge_split(edge_index, holdout_frac=0.15, seed=0)
    raw.setdefault("edge_index_views", {})[BACKGROUND_VIEW] = split.background_edge_index
    raw.setdefault("target_edge_index_views", {})[HOLDOUT_VIEW] = split.holdout_edge_index
    raw.setdefault("benchmark_target_stats", {})["static_split"] = split.stats
    print(
        "Neighbor-matching edge split: artifact had no stored static views; derived "
        "canonical seed-0 85/15 undirected-pair split in memory (source file unchanged).",
        flush=True,
    )


def configure_edge_split(
    raw: Any,
    dataset,
    kwargs: dict[str, Any],
    *,
    n_hop: int,
    sampler_kwargs: dict[str, Any],
) -> None:
    """Attach the holdout-positive sampler required by split-aware NM.

    The background sampler is already ``dataset.neighbor_sampler`` because the
    dataset loader built its graph from ``edge_view=static_background``.  This
    function fails closed when either named view is wrong or missing.
    """
    if not enabled(kwargs):
        return
    if kwargs.get("task_name") != "neighbor_matching":
        raise ValueError(
            "neighbor_matching_edge_split is only valid for task_name=neighbor_matching."
        )
    background_view = str(kwargs.get("edge_view", "default"))
    three_way = background_view == TRAIN_VIEW
    required_background = TRAIN_VIEW if three_way else BACKGROUND_VIEW
    if background_view != required_background:
        raise ValueError(
            "neighbor_matching_edge_split requires edge_view=static_background "
            "or edge_view=static_train; "
            f"got {background_view!r}."
        )
    holdout_view = str(kwargs.get("target_edge_view", TEST_VIEW if three_way else HOLDOUT_VIEW))
    required_target = TEST_VIEW if three_way else HOLDOUT_VIEW
    if holdout_view != required_target:
        raise ValueError(
            f"neighbor_matching_edge_split requires target_edge_view={required_target}; "
            f"got {holdout_view!r}."
        )
    holdout_edge_index = _target_edge_index(raw, holdout_view)
    if holdout_edge_index is None:
        raise ValueError(
            "neighbor_matching_edge_split requested, but the graph artifact has no "
            "target_edge_index_views['static_holdout']. Enrich or rebuild the artifact; "
            "falling back to the full adjacency would leak test positives."
        )
    if holdout_edge_index.dim() != 2 or holdout_edge_index.shape[0] != 2:
        raise ValueError(
            "static_holdout must have shape [2, E], got "
            f"{tuple(holdout_edge_index.shape)}."
        )

    test_graph = Data(edge_index=holdout_edge_index, num_nodes=int(dataset.graph.num_nodes))
    dataset.nm_test_neighbor_sampler = NeighborSampler(
        test_graph,
        num_hops=n_hop,
        **sampler_kwargs,
    )
    dataset.nm_holdout_neighbor_sampler = dataset.nm_test_neighbor_sampler
    if three_way:
        validation_edge_index = _target_edge_index(raw, VALIDATION_VIEW)
        if validation_edge_index is None:
            raise ValueError("three-way split has no static_validation target view")
        validation_graph = Data(
            edge_index=validation_edge_index,
            num_nodes=int(dataset.graph.num_nodes),
        )
        dataset.nm_validation_neighbor_sampler = NeighborSampler(
            validation_graph,
            num_hops=n_hop,
            **sampler_kwargs,
        )
    else:
        validation_edge_index = holdout_edge_index
        dataset.nm_validation_neighbor_sampler = dataset.nm_test_neighbor_sampler
    dataset.nm_background_edge_view = background_view
    dataset.nm_holdout_edge_view = holdout_view
    print(
        f"Neighbor-matching edge split: train/context={background_view} "
        f"({dataset.graph.edge_index.shape[1]} directed edges), "
        f"validation positives={VALIDATION_VIEW if three_way else holdout_view} "
        f"({validation_edge_index.shape[1]} directed edges), test positives={holdout_view} "
        f"({holdout_edge_index.shape[1]} directed edges)",
        flush=True,
    )


def positive_sampler_for_split(dataset, split: str, kwargs: dict[str, Any]):
    """Return the positive-edge sampler for one NM dataloader split."""
    if not enabled(kwargs) or split == "train":
        return dataset.neighbor_sampler
    attr = "nm_validation_neighbor_sampler" if split in {"val", "train_val"} else "nm_test_neighbor_sampler"
    sampler = getattr(dataset, attr, None)
    if sampler is None:
        sampler = getattr(dataset, "nm_holdout_neighbor_sampler", None)
    if sampler is None:
        raise ValueError(
            "Split-aware neighbor matching requested for val/test, but the dataset "
            "has no holdout sampler. The dataset loader must call configure_edge_split."
        )
    return sampler
