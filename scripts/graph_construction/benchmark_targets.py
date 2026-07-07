"""Shared benchmark-target helpers for retweet-graph generators.

This module centralises two additions to the graph artifact contract so every
per-dataset generator produces them identically:

1. **Node regression targets** (``node_targets`` / ``node_target_names``) — a
   fixed panel of exogenous user-profile attributes used as continuous
   node-regression targets. These are stored *separately from* ``x`` (NaN for
   missing users) so that (a) the bio-embedding feature matrix stays pure and
   passes the ``torch.isnan(x)`` generator guard, and (b) regression node splits
   can simply drop NaN nodes (``data._build_regression_node_splits`` already does
   this via ``np.isfinite``).

2. **Static link-prediction edge split** (``static_background`` /
   ``static_holdout`` views) — a seeded random split of the graph's existing
   edges. Held-out edges are *removed* from the background (message-passing) view
   so the encoder never sees the edges it is asked to predict. The split is done
   over *undirected* pairs so a held-out (u, v) can never leak via (v, u) in the
   background.

The targets are raw values (interpretable, reusable); log1p / ranking transforms
are applied downstream at eval time.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Optional

import numpy as np
import torch


# Fixed target panel + storage order. account_age_days is derived from the
# account creation timestamp; the other five are raw profile counts.
PROFILE_COUNT_FIELDS = (
    "followers_count",
    "friends_count",
    "statuses_count",
    "favourites_count",
    "listed_count",
)
PROFILE_TARGET_NAMES = (*PROFILE_COUNT_FIELDS, "account_age_days")

# Twitter timestamp formats encountered across the datasets.
_TWITTER_DATETIME_FORMATS = (
    "%a %b %d %H:%M:%S %z %Y",          # v1.1 "Wed Oct 10 20:19:24 +0000 2018"
    "%Y-%m-%d %H:%M:%S%z",              # some CSV exports
    "%Y-%m-%dT%H:%M:%S.%f%z",           # ISO 8601 with millis (TwiBot-20)
    "%Y-%m-%dT%H:%M:%S%z",              # ISO 8601
    "%Y-%m-%d %H:%M:%S",                 # naive
    "%Y-%m-%d",                          # date only
)


def parse_twitter_datetime(value: Any) -> Optional[datetime]:
    """Parse a Twitter created-at value into a tz-aware UTC ``datetime``.

    Accepts native datetimes, epoch seconds/millis (int/float), and the several
    string formats in ``_TWITTER_DATETIME_FORMATS``. Returns ``None`` on failure
    so callers can treat unparseable creation dates as missing.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not np.isfinite(value):
            return None
        # Heuristic: values above ~1e12 are epoch millis.
        seconds = float(value) / 1000.0 if float(value) > 1e12 else float(value)
        try:
            return datetime.fromtimestamp(seconds, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "null"}:
        return None
    for fmt in _TWITTER_DATETIME_FORMATS:
        try:
            parsed = datetime.strptime(text, fmt)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _coerce_count(value: Any) -> float:
    """Coerce a raw count field to float, mapping missing/invalid to NaN."""
    if value is None:
        return float("nan")
    if isinstance(value, bool):
        return float(value)
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(out) or out < 0:
        return float("nan")
    return out


def build_profile_node_targets(
    user_ids: Iterable[int],
    raw_by_user: Mapping[int, Mapping[str, Any]],
    *,
    reference_date: Optional[datetime] = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Build the fixed profile-target panel aligned to ``user_ids``.

    Args:
        user_ids: node order (the same stable order used to build ``x``).
        raw_by_user: maps raw user id -> a mapping that may contain the five
            count fields in ``PROFILE_COUNT_FIELDS`` plus one of
            ``account_creation`` / ``created_at`` / ``account_creation_date``.
        reference_date: date used to compute ``account_age_days``
            (``reference - creation``). Defaults to the latest observed creation
            date across users (approx. data-collection time), recorded in stats.

    Returns:
        (node_targets, stats) where ``node_targets`` maps each name in
        ``PROFILE_TARGET_NAMES`` to a float32 tensor of shape [num_nodes] with
        NaN for missing values.
    """
    user_ids = [int(u) for u in user_ids]
    n = len(user_ids)
    counts = {name: np.full(n, np.nan, dtype=np.float32) for name in PROFILE_COUNT_FIELDS}
    creation_dt: list[Optional[datetime]] = [None] * n

    creation_keys = ("account_creation", "created_at", "account_creation_date", "creation_date")
    matched = 0
    for i, uid in enumerate(user_ids):
        rec = raw_by_user.get(uid)
        if rec is None:
            continue
        matched += 1
        for name in PROFILE_COUNT_FIELDS:
            if name in rec:
                counts[name][i] = _coerce_count(rec[name])
        for key in creation_keys:
            if key in rec and rec[key] is not None:
                creation_dt[i] = parse_twitter_datetime(rec[key])
                break

    parsed_creation = [dt for dt in creation_dt if dt is not None]
    if reference_date is None:
        reference_date = max(parsed_creation) if parsed_creation else datetime.now(timezone.utc)
    if reference_date.tzinfo is None:
        reference_date = reference_date.replace(tzinfo=timezone.utc)

    age_days = np.full(n, np.nan, dtype=np.float32)
    for i, dt in enumerate(creation_dt):
        if dt is None:
            continue
        delta = (reference_date - dt).total_seconds() / 86400.0
        age_days[i] = float(delta) if delta >= 0 else float("nan")

    node_targets: dict[str, torch.Tensor] = {
        name: torch.from_numpy(counts[name]) for name in PROFILE_COUNT_FIELDS
    }
    node_targets["account_age_days"] = torch.from_numpy(age_days)

    stats = {
        "num_nodes": n,
        "profile_matched_users": matched,
        "reference_date": reference_date.isoformat(),
        "coverage": {
            name: int(np.isfinite(node_targets[name].numpy()).sum())
            for name in PROFILE_TARGET_NAMES
        },
    }
    return node_targets, stats


@dataclass(frozen=True)
class StaticEdgeSplit:
    """Result of :func:`build_static_edge_split`.

    ``background_mask`` is a boolean tensor over the *original* edge columns
    (True = kept in the message-passing graph). ``holdout_edge_index`` are the
    held-out positive target edges for static link prediction.
    """

    background_mask: torch.Tensor
    background_edge_index: torch.Tensor
    holdout_edge_index: torch.Tensor
    stats: dict[str, Any]


def build_static_edge_split(
    edge_index: torch.Tensor,
    *,
    holdout_frac: float = 0.15,
    seed: int = 0,
) -> StaticEdgeSplit:
    """Split existing edges into background / held-out views over undirected pairs.

    The split is performed on *unordered* node pairs, so both directions of a
    reciprocated edge are assigned together — a held-out (u, v) is never
    recoverable from (v, u) left in the background graph. All directed edges whose
    undirected pair is held out are removed from the background view.

    Args:
        edge_index: LongTensor [2, E] directed edges.
        holdout_frac: fraction of undirected pairs to hold out (0 < f < 1).
        seed: RNG seed for a reproducible split.
    """
    if edge_index.dim() != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E]")
    if not 0.0 < holdout_frac < 1.0:
        raise ValueError(f"holdout_frac must be in (0, 1), got {holdout_frac}")

    ei = edge_index.long().cpu().numpy()
    src, dst = ei[0], ei[1]
    lo = np.minimum(src, dst)
    hi = np.maximum(src, dst)
    # Canonical undirected pair key -> unique id per pair.
    pair_keys = np.stack([lo, hi], axis=1)
    _, inverse, counts = np.unique(pair_keys, axis=0, return_inverse=True, return_counts=True)
    num_pairs = int(counts.shape[0])

    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_pairs)
    n_holdout = max(1, int(round(num_pairs * holdout_frac)))
    n_holdout = min(n_holdout, num_pairs - 1)  # never hold out every pair
    holdout_pairs = np.zeros(num_pairs, dtype=bool)
    holdout_pairs[perm[:n_holdout]] = True

    edge_is_holdout = holdout_pairs[inverse]
    background_mask_np = ~edge_is_holdout

    background_mask = torch.from_numpy(background_mask_np)
    background_edge_index = edge_index[:, background_mask]
    holdout_edge_index = edge_index[:, torch.from_numpy(edge_is_holdout)]

    stats = {
        "total_edges": int(ei.shape[1]),
        "undirected_pairs": num_pairs,
        "holdout_pairs": int(n_holdout),
        "holdout_frac_requested": holdout_frac,
        "background_edges": int(background_edge_index.shape[1]),
        "holdout_edges": int(holdout_edge_index.shape[1]),
        "seed": seed,
    }
    return StaticEdgeSplit(
        background_mask=background_mask,
        background_edge_index=background_edge_index,
        holdout_edge_index=holdout_edge_index,
        stats=stats,
    )


def attach_benchmark_targets(
    graph_obj: dict[str, Any],
    *,
    node_targets: Optional[Mapping[str, torch.Tensor]] = None,
    static_split: Optional[StaticEdgeSplit] = None,
    edge_attr: Optional[torch.Tensor] = None,
    edge_attr_feature_names: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Attach node-regression targets and static-LP edge views to ``graph_obj``.

    Mutates and returns the generator's artifact dict, creating the ``*_views``
    sub-dicts if the generator did not already build temporal views. This keeps
    each generator's call site to a single line and guarantees an identical
    on-disk contract across datasets.
    """
    if node_targets is not None:
        graph_obj["node_targets"] = {k: v for k, v in node_targets.items()}
        graph_obj["node_target_names"] = list(PROFILE_TARGET_NAMES)

    if static_split is not None:
        eiv = graph_obj.setdefault("edge_index_views", {})
        tev = graph_obj.setdefault("target_edge_index_views", {})
        eiv["static_background"] = static_split.background_edge_index
        tev["static_holdout"] = static_split.holdout_edge_index
        if edge_attr is not None:
            eav = graph_obj.setdefault("edge_attr_views", {})
            eafv = graph_obj.setdefault("edge_attr_feature_names_views", {})
            eav["static_background"] = edge_attr[static_split.background_mask]
            eafv["static_background"] = list(edge_attr_feature_names or [])
        graph_obj.setdefault("static_split_stats", static_split.stats)

    return graph_obj


__all__ = [
    "PROFILE_COUNT_FIELDS",
    "PROFILE_TARGET_NAMES",
    "parse_twitter_datetime",
    "build_profile_node_targets",
    "StaticEdgeSplit",
    "build_static_edge_split",
    "attach_benchmark_targets",
]
