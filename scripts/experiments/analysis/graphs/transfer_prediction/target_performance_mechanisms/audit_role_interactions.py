"""Re-express the completed removal factorial as conditional role effects."""
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def contrasts(rows):
    groups = {}
    variants = {"baseline", "edges_query", "edges_support", "edges_both"}
    for row in rows:
        if row["variant"] not in variants:
            continue
        key = row["stream"], row["target"], row["source"]
        group = groups.setdefault(key, {})
        if row["variant"] in group:
            raise ValueError("duplicate factorial cell")
        group[row["variant"]] = row
    output = []
    for (stream, target, source), group in sorted(groups.items()):
        if set(group) != variants:
            raise ValueError("incomplete factorial")
        for field in ("checkpoint", "weights_sha256", "episode_fingerprint", "queries", "episodes"):
            if len({r[field] for r in group.values()}) != 1:
                raise ValueError(f"factorial changed {field}")
        r = {"stream": stream, "target": target, "source": source,
             "foreign_source": group["baseline"]["foreign_source"]}
        for metric in ("roc_auc", "accuracy", "nll"):
            b, q, s, both = (float(group[v][metric]) for v in ("baseline", "edges_query", "edges_support", "edges_both"))
            r.update({f"{metric}_baseline": b, f"{metric}_query_removed": q,
                      f"{metric}_support_removed": s, f"{metric}_both_removed": both,
                      f"{metric}_query_effect_support_intact": q-b,
                      f"{metric}_query_effect_support_removed": both-s,
                      f"{metric}_support_effect_query_intact": s-b,
                      f"{metric}_support_effect_query_removed": both-q,
                      f"{metric}_interaction": both-q-s+b})
        output.append(r)
    return output


def main():
    path = HERE / "data/role_context_cells.csv"
    with path.open() as stream:
        rows = contrasts(list(csv.DictReader(stream)))
    if len(rows) != 90 or len({r["source"] for r in rows}) != 9 or len({r["target"] for r in rows}) != 5:
        raise ValueError("expected full nine-source, five-target, two-stream panel")
    dest = HERE / "data/role_interaction_reaudit.csv"
    with dest.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "cells": len(rows), "targets": {}}
    for target in sorted({r["target"] for r in rows}):
        target_rows = [r for r in rows if r["target"] == target and r["foreign_source"] == "True"]
        sources = sorted({r["source"] for r in target_rows})
        # These are descriptive sign counts, not inferential tests. Both streams
        # must satisfy the property for a source to enter a count.
        predicates = {
            "query_removal_harms_alone_but_helps_after_support_removal": lambda r: r["roc_auc_query_effect_support_intact"] < 0 < r["roc_auc_query_effect_support_removed"],
            "both_removals_beat_baseline_and_either_single": lambda r: r["roc_auc_both_removed"] > max(r["roc_auc_baseline"], r["roc_auc_query_removed"], r["roc_auc_support_removed"]),
            "support_removal_improves_auc": lambda r: r["roc_auc_support_effect_query_intact"] > 0,
        }
        summary["targets"][target] = {name: [source for source in sources if all(predicate(r) for r in target_rows if r["source"] == source)]
                                        for name, predicate in predicates.items()}
    (HERE / "data/role_interaction_reaudit.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
