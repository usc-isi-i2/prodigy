#!/usr/bin/env python3
"""Build conservative Facebook page-reference tables from copied pickle files.

The input tree is treated as immutable. Nodes are Facebook pages. A directed
edge A -> B is created when a post by page A contains a Facebook URL that can
be resolved locally and unambiguously to page B. Self-references are removed.

Only page profile attributes and reference provenance are retained. Post text,
captions, image text, media, statistics, and history are never written.
"""

from __future__ import annotations

import argparse
import collections
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import statistics
import urllib.parse

import pandas as pd


CONTENT_PATTERNS = {"photo", "posts", "videos", "groups", "story", "watch", "reels"}
METHOD_RANK = {"exact_post_url": 0, "unique_content_id": 1, "account_url_or_handle": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--audit-per-stratum", type=int, default=25)
    return parser.parse_args()


def normalized_url(value: object) -> str:
    if not value:
        return ""
    try:
        parsed = urllib.parse.urlparse(str(value).strip())
    except Exception:
        return ""
    host = parsed.netloc.lower()
    for prefix in ("www.", "m.", "web.", "mobile."):
        if host.startswith(prefix):
            host = host[len(prefix):]
    path = urllib.parse.unquote(parsed.path or "/")
    while "//" in path:
        path = path.replace("//", "/")
    path = path.rstrip("/").lower() or "/"
    query = urllib.parse.parse_qs(parsed.query)
    query_keys: tuple[str, ...] = ()
    if path.endswith("story.php"):
        query_keys = ("id", "story_fbid")
    elif path.endswith(("photo.php", "video.php")) or path == "/watch":
        query_keys = ("id", "fbid", "v")
    kept = [(key, query[key][0]) for key in query_keys if query.get(key)]
    suffix = urllib.parse.urlencode(kept)
    return f"{host}{path}?{suffix}" if suffix else f"{host}{path}"


def is_facebook_url(value: object) -> bool:
    try:
        host = urllib.parse.urlparse(str(value)).netloc.lower()
    except Exception:
        return False
    return host == "facebook.com" or host.endswith(".facebook.com") or host == "fb.watch" or host.endswith(".fb.watch")


def url_pattern(value: object) -> str:
    parsed = urllib.parse.urlparse(str(value))
    path = parsed.path.lower()
    if "/groups/" in path:
        return "groups"
    if "/posts/" in path:
        return "posts"
    if "/videos/" in path:
        return "videos"
    if "/reel" in path:
        return "reels"
    if "story.php" in path:
        return "story"
    if "photo" in path:
        return "photo"
    if "/watch" in path or parsed.netloc.lower().endswith("fb.watch"):
        return "watch"
    return "other"


def content_ids(value: object) -> set[str]:
    parsed = urllib.parse.urlparse(str(value or ""))
    segments = [segment for segment in parsed.path.split("/") if segment]
    lowered = [segment.lower() for segment in segments]
    result: set[str] = set()
    for marker in ("posts", "videos", "permalink", "reel", "reels"):
        if marker in lowered:
            index = lowered.index(marker)
            if index + 1 < len(segments):
                result.add(segments[index + 1])
    query = urllib.parse.parse_qs(parsed.query)
    for key in ("story_fbid", "fbid", "v"):
        result.update(item for item in query.get(key, []) if item)
    return result


def account_keys(account: dict) -> set[str]:
    result = {
        str(account.get(field) or "").strip().lower()
        for field in ("platformId", "id", "handle")
    }
    parsed = urllib.parse.urlparse(str(account.get("url") or ""))
    segments = [segment.lower() for segment in parsed.path.split("/") if segment]
    if len(segments) >= 2 and segments[0] == "groups":
        result.add(segments[1])
    elif segments:
        result.add(segments[0])
    query = urllib.parse.parse_qs(parsed.query)
    result.update(str(item).lower() for item in query.get("id", []) if item)
    return {item for item in result if item}


def owner_key_from_url(value: object) -> str:
    parsed = urllib.parse.urlparse(str(value))
    segments = [segment for segment in parsed.path.split("/") if segment]
    lowered = [segment.lower() for segment in segments]
    query = urllib.parse.parse_qs(parsed.query)
    if query.get("id"):
        return str(query["id"][0]).lower()
    if len(segments) >= 2 and lowered[0] == "groups":
        return segments[1].lower()
    reserved = {
        "watch", "share", "sharer", "story.php", "photo.php", "reel", "reels",
        "video.php", "l.php", "dialog", "plugins",
    }
    return lowered[0] if segments and lowered[0] not in reserved else ""


def optional_nonnegative_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


def update_metadata(existing: dict | None, account: dict, observed_at: str) -> dict:
    if existing is None:
        existing = {
            "account_id": str(account.get("platformId") or account.get("id") or ""),
            "account_type": str(account.get("accountType") or ""),
            "account_name": "",
            "account_handle": "",
            "account_url": "",
            "page_description": "",
            "page_category": "",
            "page_admin_top_country": "",
            "page_created_date": "",
            "verified": None,
            "subscriber_count": None,
            "metadata_observed_at": "",
            "description_observed_at": "",
            "first_post_date": "",
            "last_post_date": "",
            "source_post_count": 0,
        }
    existing["source_post_count"] += 1
    if observed_at >= existing["metadata_observed_at"]:
        for source_key, target_key in (
            ("name", "account_name"),
            ("handle", "account_handle"),
            ("url", "account_url"),
            ("pageCategory", "page_category"),
            ("pageAdminTopCountry", "page_admin_top_country"),
            ("pageCreatedDate", "page_created_date"),
        ):
            value = str(account.get(source_key) or "").strip()
            if value:
                existing[target_key] = value
        if "verified" in account:
            existing["verified"] = bool(account.get("verified"))
        subscriber_count = optional_nonnegative_int(account.get("subscriberCount"))
        if subscriber_count is not None:
            existing["subscriber_count"] = subscriber_count
        existing["metadata_observed_at"] = observed_at
    description = str(account.get("pageDescription") or "").strip()
    if description and observed_at >= existing["description_observed_at"]:
        existing["page_description"] = description
        existing["description_observed_at"] = observed_at
    return existing


def unique_owner(values: set[str]) -> str:
    return next(iter(values)) if len(values) == 1 else ""


def resolve_target(
    url: str,
    post_url_owner: dict[str, set[str]],
    content_owner: dict[str, set[str]],
    key_owner: dict[str, set[str]],
) -> tuple[str, str, str]:
    exact = post_url_owner.get(normalized_url(url), set())
    owner = unique_owner(exact)
    if owner:
        return owner, "exact_post_url", "resolved"
    by_content: set[str] = set()
    for content_id in content_ids(url):
        by_content.update(content_owner.get(content_id, set()))
    owner = unique_owner(by_content)
    if owner:
        return owner, "unique_content_id", "resolved"
    key = owner_key_from_url(url)
    by_key = key_owner.get(key, set()) if key else set()
    owner = unique_owner(by_key)
    if owner:
        return owner, "account_url_or_handle", "resolved"
    if len(exact) > 1 or len(by_content) > 1 or len(by_key) > 1:
        return "", "ambiguous", "ambiguous"
    return "", "unresolved", "unresolved"


def add_audit_sample(
    reservoirs: dict,
    seen: collections.Counter,
    stratum: tuple[str, str, str],
    record: dict,
    limit: int,
    rng: random.Random,
) -> None:
    seen[stratum] += 1
    bucket = reservoirs.setdefault(stratum, [])
    if len(bucket) < limit:
        bucket.append(record)
        return
    replacement = rng.randrange(seen[stratum])
    if replacement < limit:
        bucket[replacement] = record


def graph_stats(events: list[dict]) -> dict:
    edge_counts = collections.Counter(
        (event["source_account_id"], event["target_account_id"])
        for event in events
    )
    nodes = {node for edge in edge_counts for node in edge}
    parent = {node: node for node in nodes}
    degree = collections.Counter()
    in_degree = collections.Counter()
    out_degree = collections.Counter()

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for source, target in edge_counts:
        union(source, target)
        degree[source] += 1
        degree[target] += 1
        out_degree[source] += 1
        in_degree[target] += 1
    sizes = sorted(collections.Counter(find(node) for node in nodes).values(), reverse=True)
    weights = list(edge_counts.values())
    return {
        "nodes": len(nodes),
        "directed_edges": len(edge_counts),
        "reference_events": len(events),
        "weak_components": len(sizes),
        "largest_component": sizes[0] if sizes else 0,
        "largest_component_fraction": round((sizes[0] if sizes else 0) / max(1, len(nodes)), 6),
        "weak_degree_ge_7": sum(value >= 7 for value in degree.values()),
        "out_degree_ge_7": sum(value >= 7 for value in out_degree.values()),
        "in_degree_ge_7": sum(value >= 7 for value in in_degree.values()),
        "edge_weight_median": statistics.median(weights) if weights else 0,
        "edge_weight_max": max(weights) if weights else 0,
    }


class IncrementalGraphStats:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.component_size: dict[str, int] = {}
        self.edge_counts: collections.Counter = collections.Counter()
        self.degree: collections.Counter = collections.Counter()
        self.in_degree: collections.Counter = collections.Counter()
        self.out_degree: collections.Counter = collections.Counter()
        self.components = 0
        self.largest_component = 0
        self.events = 0
        self.weak_degree_ge_7 = 0
        self.in_degree_ge_7 = 0
        self.out_degree_ge_7 = 0

    def _add_node(self, node: str) -> None:
        if node not in self.parent:
            self.parent[node] = node
            self.component_size[node] = 1
            self.components += 1
            self.largest_component = max(self.largest_component, 1)

    def _find(self, node: str) -> str:
        while self.parent[node] != node:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node

    def _union(self, left: str, right: str) -> None:
        left_root, right_root = self._find(left), self._find(right)
        if left_root == right_root:
            return
        if self.component_size[left_root] < self.component_size[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.component_size[left_root] += self.component_size.pop(right_root)
        self.components -= 1
        self.largest_component = max(self.largest_component, self.component_size[left_root])

    @staticmethod
    def _increment(counter: collections.Counter, key: str) -> bool:
        counter[key] += 1
        return counter[key] == 7

    def add(self, event: dict) -> None:
        self.events += 1
        source = event["source_account_id"]
        target = event["target_account_id"]
        edge = (source, target)
        if self.edge_counts[edge] == 0:
            self._add_node(source)
            self._add_node(target)
            self.weak_degree_ge_7 += int(self._increment(self.degree, source))
            self.weak_degree_ge_7 += int(self._increment(self.degree, target))
            self.out_degree_ge_7 += int(self._increment(self.out_degree, source))
            self.in_degree_ge_7 += int(self._increment(self.in_degree, target))
            self._union(source, target)
        self.edge_counts[edge] += 1

    def snapshot(self) -> dict:
        nodes = len(self.parent)
        return {
            "nodes": nodes,
            "directed_edges": len(self.edge_counts),
            "reference_events": self.events,
            "weak_components": self.components,
            "largest_component": self.largest_component,
            "largest_component_fraction": round(self.largest_component / max(1, nodes), 6),
            "weak_degree_ge_7": self.weak_degree_ge_7,
            "out_degree_ge_7": self.out_degree_ge_7,
            "in_degree_ge_7": self.in_degree_ge_7,
        }


def growth_rows(events: list[dict], input_records: list[dict]) -> list[dict]:
    days = sorted({row["partition_date"] for row in input_records})
    events_by_day: dict[str, list[dict]] = collections.defaultdict(list)
    for event in events:
        events_by_day[event["source_partition_date"]].append(event)
    primary = IncrementalGraphStats()
    content = IncrementalGraphStats()
    file_count = byte_count = record_count = 0
    inputs_by_day: dict[str, list[dict]] = collections.defaultdict(list)
    for record in input_records:
        inputs_by_day[record["partition_date"]].append(record)
    rows = []
    for day in days:
        for record in inputs_by_day[day]:
            file_count += 1
            byte_count += record["size_bytes"]
            record_count += record["records"]
        for event in events_by_day.get(day, []):
            primary.add(event)
            if event["is_content_reference"]:
                content.add(event)
        row = {
            "cutoff_date": day,
            "input_files": file_count,
            "input_bytes": byte_count,
            "input_records": record_count,
        }
        row.update({f"primary_{key}": value for key, value in primary.snapshot().items()})
        row.update({f"content_{key}": value for key, value in content.snapshot().items()})
        rows.append(row)
    return rows


def write_parquet(records: list[dict], path: Path) -> None:
    pd.DataFrame.from_records(records).to_parquet(
        path, index=False, engine="pyarrow", compression="zstd"
    )


def main() -> None:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not input_root.is_dir():
        raise NotADirectoryError(input_root)
    if input_root == output_root or input_root in output_root.parents:
        raise ValueError("Output must not be inside the immutable input tree")
    files = sorted(input_root.rglob("*.pkl"))
    if not files:
        raise FileNotFoundError(f"No .pkl files found recursively under {input_root}")
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_root}")
    building_root = output_root.with_name(output_root.name + f".building.{os.getpid()}")
    if building_root.exists():
        raise FileExistsError(f"Refusing to reuse temporary output: {building_root}")
    building_root.mkdir(parents=True)

    account_types: dict[str, str] = {}
    account_metadata: dict[str, dict] = {}
    page_profiles: dict[str, dict] = {}
    post_url_owner: dict[str, set[str]] = collections.defaultdict(set)
    content_owner: dict[str, set[str]] = collections.defaultdict(set)
    key_owner: dict[str, set[str]] = collections.defaultdict(set)
    candidates: list[dict] = []
    input_records: list[dict] = []
    counters = collections.Counter()
    seen_posts: set[str] = set()

    for file_index, path in enumerate(files, 1):
        relative_path = path.relative_to(input_root)
        partition_date = relative_path.parts[0] if len(relative_path.parts) > 1 else ""
        with path.open("rb") as handle:
            rows = pickle.load(handle)
        input_records.append({
            "relative_path": str(relative_path),
            "partition_date": partition_date,
            "size_bytes": path.stat().st_size,
            "records": len(rows),
        })
        counters["records_total"] += len(rows)
        for row in rows:
            platform = str(row.get("platform") or "")
            counters[f"platform::{platform or '<empty>'}"] += 1
            if platform != "Facebook":
                continue
            counters["facebook_records"] += 1
            account = row.get("account") or {}
            source = str(account.get("platformId") or account.get("id") or "").strip()
            if not source:
                counters["facebook_records_missing_account"] += 1
                continue
            account_type = str(account.get("accountType") or "<empty>")
            account_types[source] = account_type
            observed_at = str(row.get("updated") or row.get("date") or "")
            metadata = update_metadata(account_metadata.get(source), account, observed_at)
            post_date = str(row.get("date") or "")
            if post_date:
                if not metadata["first_post_date"] or post_date < metadata["first_post_date"]:
                    metadata["first_post_date"] = post_date
                if post_date > metadata["last_post_date"]:
                    metadata["last_post_date"] = post_date
            account_metadata[source] = metadata
            if account_type == "facebook_page":
                page_profiles[source] = metadata
            for key in account_keys(account):
                key_owner[key].add(source)
            post_url = normalized_url(row.get("postUrl"))
            post_key = str(row.get("platformId") or row.get("id") or post_url)
            if post_key in seen_posts:
                counters["duplicate_post_records"] += 1
            else:
                seen_posts.add(post_key)
            if post_url:
                post_url_owner[post_url].add(source)
            row_content_ids = content_ids(row.get("postUrl"))
            row_content_ids.add(str(row.get("id") or "").strip())
            row_content_ids.add(str(row.get("platformId") or "").split("_")[-1].strip())
            for content_id in row_content_ids:
                if content_id:
                    content_owner[content_id].add(source)
            if account_type != "facebook_page":
                continue
            row_has_facebook_link = False
            for item in row.get("expandedLinks") or []:
                if not isinstance(item, dict):
                    continue
                url = str(item.get("expanded") or item.get("original") or "")
                if not is_facebook_url(url):
                    continue
                row_has_facebook_link = True
                counters["page_source_facebook_links"] += 1
                candidates.append({
                    "source_post_id": post_key,
                    "source_account_id": source,
                    "event_date": post_date,
                    "source_partition_date": partition_date or post_date[:10],
                    "source_file": str(relative_path),
                    "target_url": url,
                    "normalized_target_url": normalized_url(url),
                    "url_pattern": url_pattern(url),
                })
            counters["page_source_rows_with_facebook_links"] += int(row_has_facebook_link)
        del rows
        gc.collect()
        if file_index % 10 == 0 or file_index == len(files):
            print(f"[progress] indexed {file_index}/{len(files)} files", flush=True)

    resolution_counts = collections.Counter()
    audit_seen = collections.Counter()
    audit_reservoirs: dict[tuple[str, str, str], list[dict]] = {}
    rng = random.Random(0)
    best_event: dict[tuple[str, str, str], dict] = {}
    for candidate_index, candidate in enumerate(candidates, 1):
        target, method, status = resolve_target(
            candidate["target_url"], post_url_owner, content_owner, key_owner
        )
        target_type = account_types.get(target, "") if target else ""
        if status != "resolved":
            outcome = status
        elif target == candidate["source_account_id"]:
            outcome = "self"
        elif target_type == "facebook_page":
            outcome = "cross_page"
        else:
            outcome = "cross_nonpage"
        stratum = (candidate["url_pattern"], method, outcome)
        resolution_counts[stratum] += 1
        source_meta = account_metadata.get(candidate["source_account_id"], {})
        target_meta = account_metadata.get(target, {}) if target else {}
        add_audit_sample(
            audit_reservoirs,
            audit_seen,
            stratum,
            {
                **candidate,
                "resolution_method": method,
                "outcome": outcome,
                "target_account_id": target,
                "target_account_type": target_type,
                "source_account_name": source_meta.get("account_name", ""),
                "source_account_url": source_meta.get("account_url", ""),
                "target_account_name": target_meta.get("account_name", ""),
                "target_account_handle": target_meta.get("account_handle", ""),
                "target_account_url": target_meta.get("account_url", ""),
                "extracted_owner_key": owner_key_from_url(candidate["target_url"]),
                "extracted_content_ids": json.dumps(sorted(content_ids(candidate["target_url"]))),
            },
            args.audit_per_stratum,
            rng,
        )
        if outcome == "cross_page":
            event = {
                **candidate,
                "target_account_id": target,
                "resolution_method": method,
                "is_content_reference": candidate["url_pattern"] in CONTENT_PATTERNS,
            }
            event_key = (candidate["source_post_id"], candidate["source_account_id"], target)
            previous = best_event.get(event_key)
            if previous is None or METHOD_RANK[method] < METHOD_RANK[previous["resolution_method"]]:
                best_event[event_key] = event
        if candidate_index % 100000 == 0 or candidate_index == len(candidates):
            print(f"[progress] resolved {candidate_index}/{len(candidates)} candidate links", flush=True)

    events = sorted(
        best_event.values(),
        key=lambda row: (row["source_account_id"], row["target_account_id"], row["source_post_id"]),
    )
    edge_groups: dict[tuple[str, str], dict] = {}
    for event in events:
        key = (event["source_account_id"], event["target_account_id"])
        edge = edge_groups.setdefault(key, {
            "source_account_id": key[0],
            "target_account_id": key[1],
            "n_reference_posts": 0,
            "n_content_reference_posts": 0,
            "first_event_date": "",
            "last_event_date": "",
            "resolution_methods": collections.Counter(),
            "url_patterns": collections.Counter(),
        })
        edge["n_reference_posts"] += 1
        edge["n_content_reference_posts"] += int(event["is_content_reference"])
        date = event["event_date"]
        if date and (not edge["first_event_date"] or date < edge["first_event_date"]):
            edge["first_event_date"] = date
        if date and date > edge["last_event_date"]:
            edge["last_event_date"] = date
        edge["resolution_methods"][event["resolution_method"]] += 1
        edge["url_patterns"][event["url_pattern"]] += 1
    edges = []
    for key in sorted(edge_groups):
        edge = edge_groups[key]
        edge["resolution_methods_json"] = json.dumps(dict(sorted(edge.pop("resolution_methods").items())))
        edge["url_patterns_json"] = json.dumps(dict(sorted(edge.pop("url_patterns").items())))
        edges.append(edge)

    graph_nodes = {node for edge in edge_groups for node in edge}
    profiles = []
    for account_id in sorted(page_profiles):
        record = dict(page_profiles[account_id])
        record["in_primary_graph"] = account_id in graph_nodes
        profiles.append(record)
    audits = [record for key in sorted(audit_reservoirs) for record in audit_reservoirs[key]]
    audit_counts = [
        {
            "url_pattern": pattern,
            "resolution_method": method,
            "outcome": outcome,
            "candidate_links": count,
            "audit_rows": len(audit_reservoirs.get((pattern, method, outcome), [])),
        }
        for (pattern, method, outcome), count in sorted(resolution_counts.items())
    ]
    growth = growth_rows(events, input_records)

    write_parquet(input_records, building_root / "input_files.parquet")
    write_parquet(profiles, building_root / "page_profiles.parquet")
    write_parquet(events, building_root / "page_reference_events.parquet")
    write_parquet(edges, building_root / "page_reference_edges.parquet")
    write_parquet(audits, building_root / "resolution_audit.parquet")
    write_parquet(audit_counts, building_root / "resolution_counts.parquet")
    write_parquet(growth, building_root / "growth_by_cutoff.parquet")

    content_events = [event for event in events if event["is_content_reference"]]
    summary = {
        "input_root": str(input_root),
        "input_files": len(files),
        "input_bytes": sum(row["size_bytes"] for row in input_records),
        "records_total": counters["records_total"],
        "facebook_records": counters["facebook_records"],
        "unique_facebook_posts": len(seen_posts),
        "duplicate_post_records": counters["duplicate_post_records"],
        "unique_page_profiles": len(profiles),
        "page_profiles_with_description": sum(bool(row["page_description"]) for row in profiles),
        "page_source_rows_with_facebook_links": counters["page_source_rows_with_facebook_links"],
        "page_source_facebook_links": counters["page_source_facebook_links"],
        "candidate_resolution_counts": {
            "|".join(key): value for key, value in sorted(resolution_counts.items())
        },
        "primary_page_reference_graph": graph_stats(events),
        "content_only_page_reference_graph": graph_stats(content_events),
        "growth_by_cutoff": growth,
        "outputs": {
            "page_profiles": len(profiles),
            "page_reference_events": len(events),
            "page_reference_edges": len(edges),
            "resolution_audit": len(audits),
            "resolution_count_strata": len(audit_counts),
            "growth_cutoffs": len(growth),
        },
        "construction": {
            "nodes": "Facebook accounts with accountType=facebook_page",
            "edge": "A directed edge A->B exists when a post by page A contains a locally and unambiguously resolved Facebook reference to page B",
            "edge_weight": "number of distinct source posts from A referencing B",
            "self_links_removed": True,
            "post_text_retained": False,
            "growth_note": "Daily growth uses resolution indices from the full input window and is for cutoff sizing; rerun on the selected final window for the final artifact.",
            "audit_seed": 0,
            "audit_per_stratum": args.audit_per_stratum,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }
    with (building_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    (building_root / "README.md").write_text(
        "# Facebook page-reference tables\n\n"
        "Inputs are copied pickle files; the source tree is treated as immutable. "
        "No post message, caption, description, image text, media URL, statistics, "
        "or engagement history is retained. `page_description` refers only to the "
        "page-profile biography field.\n\n"
        "- `page_profiles.parquet`: deduplicated page profile metadata and targets.\n"
        "- `page_reference_events.parquet`: distinct source-post to target-page references.\n"
        "- `page_reference_edges.parquet`: directed page-to-page edges.\n"
        "- `growth_by_cutoff.parquet`: cumulative graph size by input date.\n"
        "- `resolution_audit.parquet`: deterministic stratified precision-audit sample.\n"
        "- `resolution_counts.parquet`: population resolution counts by stratum.\n"
        "- `input_files.parquet`: relative input paths, sizes, and record counts.\n",
        encoding="utf-8",
    )
    building_root.rename(output_root)
    print(json.dumps(summary["primary_page_reference_graph"], sort_keys=True), flush=True)
    print(f"[done] wrote {output_root}", flush=True)


if __name__ == "__main__":
    main()
