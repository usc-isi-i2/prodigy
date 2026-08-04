"""Deterministic node selection for Facebook page-reference artifacts."""

from __future__ import annotations

import pyarrow as pa


def select_page_nodes(
    structural_node_ids: set[str],
    profiles: pa.Table,
    target_node_count: int = 0,
) -> tuple[list[str], set[str]]:
    """Keep all edge participants and optionally add active page-profile isolates."""
    structural = {str(node_id) for node_id in structural_node_ids}
    requested = int(target_node_count)
    if requested < 0:
        raise ValueError("target_node_count must be non-negative")
    if requested and requested < len(structural):
        raise ValueError(
            f"target_node_count={requested} is below {len(structural)} structural nodes"
        )
    selected = set(structural)
    if requested and requested > len(selected):
        candidates = []
        for row in profiles.select(["account_id", "source_post_count"]).to_pylist():
            account_id = str(row["account_id"])
            if account_id in selected:
                continue
            candidates.append((-(int(row.get("source_post_count") or 0)), account_id))
        candidates.sort()
        needed = requested - len(selected)
        if len(candidates) < needed:
            raise ValueError(
                f"Only {len(selected) + len(candidates)} page profiles are available; "
                f"cannot select {requested} nodes"
            )
        selected.update(account_id for _, account_id in candidates[:needed])
    return sorted(selected), structural


__all__ = ["select_page_nodes"]
