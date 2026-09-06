"""Role-specific identity conflicts in complete, actually consumed NM streams.

No graph loading, resampling, model execution, or raw profile text. The center-
only oracle bounds do not constrain models with occurrence-specific context.
"""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np


def digest(values):
    return hashlib.sha256(np.asarray(values, dtype=np.int64).tobytes()).hexdigest()


def episode_stats(members, roles, context):
    members = np.asarray(members, dtype=np.int64)
    roles = np.asarray(roles, dtype=bool)
    context = np.asarray(context, dtype=np.int64)
    if members.ndim != 2 or context.shape != members.shape or members.shape[1] != len(roles):
        raise ValueError("invalid episode dimensions")
    if not roles.any() or roles.all() or (context < 1).any():
        raise ValueError("invalid roles/context")
    if any(len(set(row)) != len(row) for row in members.tolist()):
        raise ValueError("within-class repeated identity")
    support = Counter(members[:, ~roles].ravel().tolist())
    query = Counter(members[:, roles].ravel().tolist())
    all_ids = support + query
    # With no within-class repeats, every repeated identity has different labels.
    ns, nq, nm = sum(support.values()), sum(query.values()), members.size
    overlap = set(support) & set(query)
    qs = sum(query[n] for n in overlap)
    qq = sum(c for c in query.values() if c > 1)
    sq = sum(support[n] for n in overlap)
    ss = sum(c for c in support.values() if c > 1)
    qcontexts = context[:, roles]
    q_ids = members[:, roles]
    colliding_q = np.isin(q_ids, list(overlap))
    out = {
        "episodes": 1, "support_positions": ns, "query_positions": nq,
        "member_positions": nm, "duplicate_positions": nm - len(all_ids),
        "query_also_wrong_class_support_positions": qs,
        "query_multilabel_query_positions": qq,
        "support_multilabel_support_positions": ss,
        "support_also_wrong_class_query_positions": sq,
        "identity_only_query_oracle_correct": len(query),
        "identity_only_query_ce_floor_sum": sum(c * math.log(c) for c in query.values()),
        "query_context_nodes_sum": int(qcontexts.sum()),
        "support_context_nodes_sum": int(context[:, ~roles].sum()),
        "colliding_query_context_nodes_sum": int(qcontexts[colliding_q].sum()),
        "query_without_context_positions": int((qcontexts == 1).sum()),
        "episodes_with_query_support_conflict": int(qs > 0),
        "episodes_with_query_query_conflict": int(qq > 0),
    }
    examples = {}
    for kind, eligible in (("query_support", overlap), ("query_query", {n for n, c in query.items() if c > 1})):
        if eligible:
            node = min(eligible)
            examples[kind] = {"center_node_id": node, "occurrences": [
                {"local_class": int(c), "member_slot": int(j), "role": "query" if roles[j] else "support",
                 "context_node_count_including_center": int(context[c, j])}
                for c, j in zip(*np.where(members == node))]}
    return out, examples


def normalized(total):
    n, s, q = total["member_positions"], total["support_positions"], total["query_positions"]
    e, c = total["episodes"], total["query_also_wrong_class_support_positions"]
    return {**total,
        "cross_class_duplicate_fraction": total["duplicate_positions"] / n,
        "query_also_wrong_class_support_fraction": c / q,
        "query_multilabel_query_fraction": total["query_multilabel_query_positions"] / q,
        "support_multilabel_support_fraction": total["support_multilabel_support_positions"] / s,
        "support_also_wrong_class_query_fraction": total["support_also_wrong_class_query_positions"] / s,
        "identity_only_query_accuracy_upper_bound": total["identity_only_query_oracle_correct"] / q,
        "identity_only_query_ce_lower_bound": total["identity_only_query_ce_floor_sum"] / q,
        "query_context_nodes_mean": total["query_context_nodes_sum"] / q,
        "support_context_nodes_mean": total["support_context_nodes_sum"] / s,
        "colliding_query_context_nodes_mean": total["colliding_query_context_nodes_sum"] / c if c else None,
        "query_without_context_fraction": total["query_without_context_positions"] / q,
        "episode_query_support_conflict_fraction": total["episodes_with_query_support_conflict"] / e,
        "episode_query_query_conflict_fraction": total["episodes_with_query_query_conflict"] / e,
    }


def audit_stream(path, params, reference, steps=2500, window_steps=500):
    total, windows = Counter(), defaultdict(Counter)
    members_seen, anchors_seen, sources = Counter(), set(), Counter()
    witnesses, witness_keys = [], set()
    roles = [0] * params["n_shots"] + [1] * params["n_query"]
    shape = (params["batch_size"], params["n_way"], len(roles))
    stream_hash = hashlib.sha256()
    count = 0
    with gzip.open(path, "rt") as handle:
        for count, line in enumerate(handle, 1):
            row = json.loads(line)
            ids, anchors = np.asarray(row["member_ids"], dtype=np.int64), np.asarray(row["anchor_ids"], dtype=np.int64)
            context = np.asarray(row["context_node_counts"], dtype=np.int64)
            if row["step"] != count or row["query_roles"] != roles or ids.shape != shape or context.shape != shape:
                raise ValueError("invalid sequence/roles/member dimensions")
            if anchors.shape != shape[:2] or any(len(set(a)) != params["n_way"] for a in anchors.tolist()):
                raise ValueError("invalid anchors")
            if len(row["source_ids"]) != params["batch_size"]:
                raise ValueError("invalid source dimensions")
            expected = (digest(anchors), digest(np.sort(ids, axis=-1)), digest(ids))
            actual = (row["anchor_sha256"], row["member_set_sha256"], row["member_order_sha256"])
            if actual != expected:
                raise ValueError("consumed identity hash mismatch")
            stream_hash.update(json.dumps(row, sort_keys=True, separators=(",", ":")).encode())
            members_seen.update(ids.ravel().tolist())
            anchors_seen.update(anchors.ravel().tolist())
            sources.update(row["source_ids"])
            window = (count - 1) // window_steps
            for ei, (part, ctx) in enumerate(zip(ids, context)):
                stats, examples = episode_stats(part, roles, ctx)
                total.update(stats)
                windows[window].update(stats)
                for kind, example in examples.items():
                    key = (window, kind)
                    if key not in witness_keys:
                        for occurrence in example["occurrences"]:
                            occurrence["anchor_id"] = int(anchors[ei, occurrence["local_class"]])
                        witnesses.append({"step": count, "episode_in_batch": ei, "window": window,
                            "kind": kind, "member_order_sha256": row["member_order_sha256"], **example})
                        witness_keys.add(key)
    if count != steps:
        raise ValueError("incomplete stream")
    actual_summary = {"steps": count, "episodes": total["episodes"], "member_positions": total["member_positions"],
        "unique_members": len(members_seen), "unique_anchors": len(anchors_seen),
        "effective_member_count": total["member_positions"] ** 2 / sum(c*c for c in members_seen.values()),
        "cross_class_duplicate_fraction": total["duplicate_positions"] / total["member_positions"],
        "context_nodes_mean": (total["query_context_nodes_sum"] + total["support_context_nodes_sum"]) / total["member_positions"],
        "source_ids": {str(k): v for k, v in sources.items()}}
    for key, expected in reference.items():
        actual = actual_summary[key]
        if isinstance(expected, float):
            if not math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-12):
                raise ValueError(f"summary mismatch: {key}")
        elif actual != expected:
            raise ValueError(f"summary mismatch: {key}")
    if len(sources) != 1:
        raise ValueError("mixed source stream")
    return {"summary": normalized(total), "windows": [{"window": k, **normalized(v)} for k, v in sorted(windows.items())],
        "witnesses": witnesses, "reference_summary_verified": actual_summary,
        "consumed_record_stream_sha256": stream_hash.hexdigest(), "top_exposed_centers": members_seen.most_common(10)}


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--member-run", type=Path, required=True)
    p.add_argument("--readout-run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("choose a new output directory")
    args.output.mkdir(parents=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    (args.output / "protocol.json").write_text(json.dumps({"revision": revision, "exploratory": True,
        "actual_complete_consumed_streams": True, "raw_bios_inspected": False,
        "bounds_apply_to": "deterministic identity-only, query-permutation-equivariant predictions within an episode; NOT full PRODIGY",
        "expected_models": 33, "steps": 2500, "window_steps": 500,
        "member_run": str(args.member_run), "readout_run": str(args.readout_run)}, indent=2))
    summaries, windows, records = [], [], []
    for campaign, root in (("member", args.member_run), ("readout", args.readout_run)):
        arms = json.loads((root / "verified/arms.json").read_text())
        manifest = json.loads((root / "manifest.json").read_text())
        if not json.loads((root / "verified/DONE.json").read_text())["research_result"]:
            raise ValueError("unverified campaign")
        plans = {job["prefix"]: (i, job) for i, job in enumerate(manifest["jobs"])}
        for arm in arms:
            if campaign == "readout" and arm["condition"] != "free":
                continue
            index, planned = plans[arm["model_id"]]
            params = json.loads((root / f"job_{index:03d}/effective_config.json").read_text())
            for k in ("exp_name", "prefix", "neighbor_sampling_source_subset", "seed", "neighbor_matching_member_policy"):
                if params[k] != planned[k]:
                    raise ValueError("effective/manifest config mismatch")
            if (params["batch_size"], params["n_way"], params["n_shots"], params["n_query"], params["epochs"] * params["dataset_len_cap"]) != (4, 30, 3, 4, 2500):
                raise ValueError("unexpected training recipe")
            path = Path(params["log_dir"]) / params["exp_name"] / "data/consumed_episodes.jsonl.gz"
            meta = {"campaign": campaign, "model_id": arm["model_id"], "source": arm["source"],
                "seed": arm["seed"], "policy": params["neighbor_matching_member_policy"]}
            result = audit_stream(path, params, arm["summary"])
            summaries.append({**meta, **result["summary"]})
            windows.extend({**meta, **r} for r in result["windows"])
            records.append({**meta, "input_path": str(path), "input_bytes": path.stat().st_size,
                **{k: v for k, v in result.items() if k not in ("summary", "windows")}})
            print(json.dumps(summaries[-1]), flush=True)
    if len(summaries) != 33 or len(windows) != 165:
        raise ValueError("incomplete campaign grid")
    for name, rows in (("models", summaries), ("windows", windows)):
        with (args.output / f"{name}.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (args.output / "records.json").write_text(json.dumps(records, indent=2))
    # Old member and newer free-readout controls have identical recorded IDs,
    # roles, anchors AND context sizes. This does not assert full context IDs.
    paired = []
    for r in records:
        if r["campaign"] != "readout" or r["source"] == "covid":
            continue
        old = [v for v in records if v["campaign"] == "member" and
               (v["source"], v["seed"], v["policy"]) == (r["source"], r["seed"], r["policy"])]
        if len(old) != 1:
            raise ValueError("missing older control")
        paired.append({"model_id": r["model_id"], "prior_model_id": old[0]["model_id"],
            "record_streams_identical": r["consumed_record_stream_sha256"] == old[0]["consumed_record_stream_sha256"]})
    (args.output / "DONE.json").write_text(json.dumps({"models": 33, "complete_streams": True,
        "total_consumed_steps": 82500, "total_episode_occurrences": 330000,
        "unique_training_streams_not_independent_models": True, "reference_summaries_verified": 33,
        "older_newer_control_record_checks": paired}, indent=2))


if __name__ == "__main__":
    main()
