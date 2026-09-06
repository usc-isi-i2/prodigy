"""Compact numeric evidence; keep bulk account text in the private paper folder."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def summarize(path):
    d = json.loads(path.read_text())
    cases = d["cases"]
    if len(cases) != 8 or set(Counter(c["stratum"] for c in cases).values()) != {2}:
        raise ValueError("expected two cases in each of four selection strata")
    if not d["verified_all_cached_features_against_graph"] or not d["verified_all_cached_center_labels"]:
        raise ValueError("unverified identities")
    if max(d["baseline_logit_max_errors"].values()) != 0:
        raise ValueError("this report requires observed bit-exact baseline reproduction")
    rows = []
    for c in cases:
        n = d["nodes"][c["node"]]
        if d["target"] != "covid_political" and not n["text_embedding_identity_verified"]:
            raise ValueError("query text was not verified against the feature shard")
        rows.append({k: c[k] for k in ("case_id", "stratum", "batch", "query_index", "episode", "node")} |
                    {"dataset_label": n["dataset_label"], "sampled_context_nodes": len(c["query_geometry"]["neighbors"]),
                     "sampled_center_indegree": c["query_geometry"]["center_indegree"],
                     "sampled_center_outdegree": c["query_geometry"]["center_outdegree"],
                     "predictions": c["predictions"], "localized_interventions": c["localized_interventions"]})
    return {"target": d["target"], "private_evidence_path": str(path.resolve()),
            "evidence_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "runtime_revision": d["revision"], "replay_root": d["root"], "episode_fingerprint": d["cache"]["episode_fingerprint"],
            "selection": d["selection"], "stratum_query_counts": d["stratum_query_counts"],
            "verified_cached_batches": len(d["cache"]["batch_sha256"]),
            "baseline_parity_checks": len(d["baseline_logit_max_errors"]), "max_baseline_logit_error": 0,
            "cases": rows}


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    rows = [summarize(args.input / f"{t}.json") for t in ("covid_political", "facebook_page_reference", "twibot20")]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(rows, f, indent=2)
    print(json.dumps({"targets": len(rows), "review_cases": sum(len(r["cases"]) for r in rows),
                      "verified_cached_batches": sum(r["verified_cached_batches"] for r in rows),
                      "baseline_parity_checks": sum(r["baseline_parity_checks"] for r in rows),
                      "new_localized_forwards": sum(len(c["localized_interventions"]) * 5 for r in rows for c in r["cases"])}))
