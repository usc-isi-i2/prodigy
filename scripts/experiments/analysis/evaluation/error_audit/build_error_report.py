#!/usr/bin/env python3
"""Enrich prediction JSONL with bios and render balanced correct/error cards.

Run once per dataset because graph node ids are dataset-local.  Raw enriched output
contains profile text and should stay under /dataMeR1 rather than being committed.
"""
from __future__ import annotations

import argparse
import html
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def read_jsonl(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if line.strip():
                    row = json.loads(line)
                    row["_source"] = path.as_posix()
                    row["_source_line"] = line_number
                    rows.append(row)
    return rows


def torch_load(path: Path):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=False)


def graph_metadata(path: Path) -> tuple[list, list, list[str], str | None]:
    blob = torch_load(path)
    get = blob.get if isinstance(blob, dict) else lambda key, default=None: getattr(blob, key, default)
    user_ids_raw = get("user_ids", [])
    handles_raw = get("handles", [])
    label_names_raw = get("label_names", [])
    user_ids = list(user_ids_raw) if user_ids_raw is not None else []
    handles = list(handles_raw) if handles_raw is not None else []
    label_names = list(label_names_raw) if label_names_raw is not None else []
    cutoff = None
    meta_path = path.with_suffix(".meta.json")
    if meta_path.exists():
        cutoff = json.loads(meta_path.read_text(encoding="utf-8")).get("graph_cutoff")
    return user_ids, handles, label_names, cutoff


def referenced_node_ids(rows: list[dict[str, Any]]) -> set[int]:
    scalar_keys = ("query_node_id", "u", "v")
    list_keys = (
        "context_node_ids", "u_context_node_ids", "v_context_node_ids",
        "support_node_ids",
    )
    nodes: set[int] = set()
    for row in rows:
        for key in scalar_keys:
            if row.get(key) is not None:
                nodes.add(int(row[key]))
        for key in list_keys:
            nodes.update(int(value) for value in row.get(key, []))
        for support in row.get("supports", []):
            nodes.add(int(support["node_id"]))
            nodes.update(int(value) for value in support.get("context_node_ids", []))
        if row.get("task") == "neighbor_matching":
            nodes.add(int(row["gt"]))
            nodes.add(int(row["prediction"]))
    return {node for node in nodes if node >= 0}


def bios_from_parquet(
    bio_root: Path, user_ids: list[Any], cutoff: str | None
) -> dict[str, str]:
    try:
        import duckdb
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("duckdb and pandas are required for --bio-root") from exc

    observations = bio_root / "user_bio_observations.parquet"
    texts = bio_root / "bio_texts.parquet"
    if not observations.exists() or not texts.exists():
        raise FileNotFoundError(
            f"Expected user_bio_observations.parquet and bio_texts.parquet under {bio_root}"
        )
    normalized = sorted({str(value) for value in user_ids if value is not None})
    if not normalized:
        return {}
    conn = duckdb.connect()
    conn.register("selected_users", pd.DataFrame({"userid": normalized}))
    if cutoff:
        query = f"""
            WITH candidates AS (
                SELECT CAST(o.userid AS VARCHAR) AS userid, o.bio_hash,
                       row_number() OVER (
                           PARTITION BY CAST(o.userid AS VARCHAR)
                           ORDER BY LEAST(
                               COALESCE(o.last_seen_at, o.first_seen_at), CAST(? AS TIMESTAMP)
                           ) DESC NULLS LAST, o.bio_hash DESC
                       ) AS rn
                FROM read_parquet(?) AS o
                INNER JOIN selected_users AS s
                    ON CAST(o.userid AS VARCHAR) = s.userid
                WHERE o.bio_hash IS NOT NULL
                  AND COALESCE(o.first_seen_at, o.last_seen_at) <= CAST(? AS TIMESTAMP)
            )
            SELECT c.userid, b.normalized_bio_text
            FROM candidates AS c
            INNER JOIN read_parquet(?) AS b ON c.bio_hash = b.bio_hash
            WHERE c.rn = 1
        """
        params = [cutoff, observations.as_posix(), cutoff, texts.as_posix()]
    else:
        query = """
            WITH candidates AS (
                SELECT CAST(o.userid AS VARCHAR) AS userid, o.bio_hash,
                       row_number() OVER (
                           PARTITION BY CAST(o.userid AS VARCHAR)
                           ORDER BY COALESCE(o.last_seen_at, o.first_seen_at)
                               DESC NULLS LAST, o.bio_hash DESC
                       ) AS rn
                FROM read_parquet(?) AS o
                INNER JOIN selected_users AS s
                    ON CAST(o.userid AS VARCHAR) = s.userid
                WHERE o.bio_hash IS NOT NULL
            )
            SELECT c.userid, b.normalized_bio_text
            FROM candidates AS c
            INNER JOIN read_parquet(?) AS b ON c.bio_hash = b.bio_hash
            WHERE c.rn = 1
        """
        params = [observations.as_posix(), texts.as_posix()]
    result = conn.execute(query, params).fetchall()
    conn.close()
    return {str(user_id): str(text) for user_id, text in result}


def bios_from_csv(
    path: Path, id_column: str, bio_column: str
) -> dict[str, str]:
    import pandas as pd

    frame = pd.read_csv(path, low_memory=False)
    ids = frame.index if id_column == "__index__" else frame[id_column]
    return {
        str(user_id): str(bio)
        for user_id, bio in zip(ids, frame[bio_column])
        if pd.notna(bio) and str(bio).strip()
    }


def profile_map(
    node_ids: set[int], user_ids: list, handles: list, bios: dict[str, str]
) -> dict[int, dict[str, Any]]:
    out = {}
    for node in node_ids:
        user_id = user_ids[node] if node < len(user_ids) else None
        handle = handles[node] if node < len(handles) else None
        if hasattr(user_id, "item"):
            user_id = user_id.item()
        if hasattr(handle, "item"):
            handle = handle.item()
        out[node] = {
            "node_id": node,
            "user_id": user_id,
            "handle": handle,
            "bio": bios.get(str(user_id), "") if user_id is not None else "",
        }
    return out


def enrich(row: dict[str, Any], profiles: dict[int, dict], label_names: list[str]) -> dict:
    row = dict(row)
    get_profile = lambda node: profiles.get(int(node), {"node_id": int(node), "bio": ""})
    if "query_node_id" in row:
        row["query_profile"] = get_profile(row["query_node_id"])
    if "context_node_ids" in row:
        row["context_profiles"] = [get_profile(node) for node in row["context_node_ids"]]
    if row.get("task") == "neighbor_matching":
        row["gt_anchor_profile"] = get_profile(row["gt"])
        row["prediction_anchor_profile"] = get_profile(row["prediction"])
    if row.get("task") == "classification" and label_names:
        for key, dest in (("gt", "gt_label_name"), ("prediction", "prediction_label_name")):
            value = int(row[key])
            row[dest] = label_names[value] if 0 <= value < len(label_names) else str(value)
    if "supports" in row:
        enriched_supports = []
        for support in row["supports"]:
            item = dict(support)
            item["profile"] = get_profile(item["node_id"])
            enriched_supports.append(item)
        row["supports"] = enriched_supports
    if "support_node_ids" in row:
        row["support_profiles"] = [
            {**get_profile(node), "target": float(target)}
            for node, target in zip(row["support_node_ids"], row.get("support_targets", []))
        ]
    for endpoint in ("u", "v"):
        if endpoint in row:
            row[f"{endpoint}_profile"] = get_profile(row[endpoint])
            row[f"{endpoint}_context_profiles"] = [
                get_profile(node) for node in row.get(f"{endpoint}_context_node_ids", [])
            ]
    return row


def assign_group(rows: list[dict[str, Any]]) -> None:
    regression = [row for row in rows if row.get("task") == "regression"]
    by_panel: dict[tuple, list[dict]] = {}
    for row in regression:
        key = (row.get("model"), row.get("dataset"), row.get("target"), row.get("alpha"))
        by_panel.setdefault(key, []).append(row)
    for panel in by_panel.values():
        errors = np.asarray([float(row["absolute_error"]) for row in panel])
        lo, hi = (float(value) for value in np.quantile(errors, [0.2, 0.8]))
        for row in panel:
            error = float(row["absolute_error"])
            row["audit_group"] = "low_error" if error <= lo else "high_error" if error >= hi else "middle_error"
            row["low_error_cutoff"] = lo
            row["high_error_cutoff"] = hi
    for row in rows:
        if row.get("task") != "regression":
            row["audit_group"] = "correct" if row.get("correct") else "incorrect"


def confidence(row: dict[str, Any]) -> float:
    if row.get("task") == "static_link_prediction":
        return abs(float(row["oriented_score"]) - float(row["decision_threshold"]))
    if row.get("task") == "regression":
        return float(row["absolute_error"])
    return float(row.get("confidence", 0.0))


def balanced_sample(rows: list[dict[str, Any]], per_group: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    selected = []
    groups = sorted({row["audit_group"] for row in rows if row["audit_group"] != "middle_error"})
    for group in groups:
        candidates = [row for row in rows if row["audit_group"] == group]
        candidates.sort(key=confidence, reverse=(group in {"incorrect", "high_error"}))
        n_hard = min(len(candidates), (per_group + 1) // 2)
        chosen = candidates[:n_hard]
        remaining = candidates[n_hard:]
        rng.shuffle(remaining)
        chosen.extend(remaining[:max(0, per_group - len(chosen))])
        selected.extend(chosen)
    return selected


def profile_html(profile: dict[str, Any]) -> str:
    ident = profile.get("handle") or profile.get("user_id") or profile.get("node_id")
    bio = profile.get("bio") or "<missing bio>"
    return f"<div class='profile'><b>{html.escape(str(ident))}</b><p>{html.escape(str(bio))}</p></div>"


def render_card(row: dict[str, Any]) -> str:
    task = row.get("task", "")
    group = row.get("audit_group", "")
    if task == "regression":
        outcome = f"GT {row['gt']:.4g} · prediction {row['prediction']:.4g} · |error| {row['absolute_error']:.4g}"
    elif task == "static_link_prediction":
        outcome = f"GT {row['gt']} · prediction {row['prediction']} · {row.get('error_type', '')}"
    else:
        gt = row.get("gt_label_name", row.get("gt"))
        pred = row.get("prediction_label_name", row.get("prediction"))
        outcome = f"GT {gt} · prediction {pred} · confidence {row.get('confidence', 0):.3f}"
    profiles = []
    if "query_profile" in row:
        profiles.append("<h4>Query</h4>" + profile_html(row["query_profile"]))
        profiles.append("<h4>Sampled neighbours</h4>" + "".join(
            profile_html(profile) for profile in row.get("context_profiles", [])
        ))
    if task == "static_link_prediction":
        profiles.append("<h4>Endpoint U</h4>" + profile_html(row["u_profile"]))
        profiles.append("<h4>Endpoint V</h4>" + profile_html(row["v_profile"]))
    supports = row.get("supports") or row.get("support_profiles") or []
    support_html = ""
    if supports:
        support_profiles = [item.get("profile", item) for item in supports[:6]]
        support_html = "<h4>Episode supports</h4>" + "".join(profile_html(p) for p in support_profiles)
    return (
        f"<article class='card {html.escape(group)}'><h3>{html.escape(task)} · {html.escape(group)}</h3>"
        f"<div class='outcome'>{html.escape(outcome)}</div>{''.join(profiles)}{support_html}</article>"
    )


def render_html(rows: list[dict[str, Any]], title: str) -> str:
    counts = Counter(row["audit_group"] for row in rows)
    cards = "".join(render_card(row) for row in rows)
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>
<style>
body{{font:15px system-ui;margin:2rem;background:#f5f5f2;color:#222}} .summary{{margin-bottom:1rem}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:1rem}}
.card{{background:white;border:1px solid #ccc;border-radius:10px;padding:1rem}}
.incorrect,.high_error{{border-left:6px solid #b42318}} .correct,.low_error{{border-left:6px solid #16803c}}
.profile{{border-top:1px solid #eee;padding:.4rem 0}} .profile p{{margin:.2rem 0;white-space:pre-wrap}}
.outcome{{font-weight:600;margin-bottom:.7rem}} h4{{margin:.8rem 0 .2rem}}
</style></head><body><h1>{html.escape(title)}</h1><div class='summary'>{html.escape(str(dict(counts)))}</div>
<div class='grid'>{cards}</div></body></html>"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", action="append", required=True, type=Path)
    ap.add_argument("--graph", required=True, type=Path)
    ap.add_argument("--bio-root", type=Path)
    ap.add_argument("--profile-csv", type=Path)
    ap.add_argument("--profile-id-column", default="__index__")
    ap.add_argument("--profile-bio-column", default="description")
    ap.add_argument("--model", default="")
    ap.add_argument("--target", default="")
    ap.add_argument("--per-group", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--title", default="Prediction error audit")
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    rows = read_jsonl(args.predictions)
    if args.model:
        rows = [row for row in rows if row.get("model", args.model) == args.model]
    if args.target:
        rows = [row for row in rows if row.get("target") == args.target]
    if not rows:
        raise ValueError("No prediction records remain after filtering")

    user_ids, handles, label_names, cutoff = graph_metadata(args.graph)
    nodes = referenced_node_ids(rows)
    selected_user_ids = [user_ids[node] for node in nodes if node < len(user_ids)]
    if args.bio_root:
        bios = bios_from_parquet(args.bio_root, selected_user_ids, cutoff)
    elif args.profile_csv:
        bios = bios_from_csv(
            args.profile_csv, args.profile_id_column, args.profile_bio_column
        )
    else:
        bios = {}
    profiles = profile_map(nodes, user_ids, handles, bios)
    enriched = [enrich(row, profiles, label_names) for row in rows]
    assign_group(enriched)
    selected = balanced_sample(enriched, args.per_group, args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    enriched_path = args.out_dir / "enriched_predictions.jsonl"
    with enriched_path.open("w", encoding="utf-8") as handle:
        for row in enriched:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    (args.out_dir / "report.html").write_text(
        render_html(selected, args.title), encoding="utf-8"
    )
    summary = {
        "n_records": len(enriched),
        "n_selected": len(selected),
        "groups": dict(Counter(row["audit_group"] for row in enriched)),
        "bios_found": sum(bool(profile.get("bio")) for profile in profiles.values()),
        "profiles": len(profiles),
        "graph": args.graph.as_posix(),
        "prediction_files": [path.as_posix() for path in args.predictions],
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
