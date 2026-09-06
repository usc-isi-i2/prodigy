"""Validate fresh replay and keep cell-level proximity distinct from query effects."""
import json
from pathlib import Path

import pandas as pd


def main():
    root = Path(__file__).parent / "data"
    old = pd.read_csv(root / "replay_cells.csv")
    fresh = pd.DataFrame([json.loads(line) for path in sorted((root / "fresh_replay").glob("*.jsonl"))
                          for line in path.read_text().splitlines()])
    if len(fresh) != 850 or fresh.duplicated(["dataset", "model_id", "decoder"]).any():
        raise ValueError("incomplete or duplicated fresh replay")
    baseline = old[old.variant == "baseline"]
    joined = fresh.merge(baseline[["dataset", "model_id", "decoder", "weights_sha256", "episode_fingerprint"]],
                         on=["dataset", "model_id", "decoder"], suffixes=("", "_original"), validate="one_to_one")
    if len(joined) != len(fresh) or not (joined.weights_sha256 == joined.weights_sha256_original).all():
        raise ValueError("fresh stream changed model weights")
    if (joined.episode_fingerprint == joined.episode_fingerprint_original).any():
        raise ValueError("fresh stream did not change episode fingerprints")
    for target, group in fresh.groupby("dataset"):
        if len(group) != 170 or group.episode_fingerprint.nunique() != 1 or set(group.episodes) != {128}:
            raise ValueError(f"incomplete/mixed fresh target {target}")
        for decoder, raw in group[group.decoder.str.startswith("raw_")].groupby("decoder"):
            if raw.roc_auc.max() - raw.roc_auc.min() > 1e-8:
                raise ValueError("raw probe depends on trained checkpoint")
    fresh.assign(sources=fresh.sources.map(json.dumps)).to_csv(root / "fresh_replay_cells.csv", index=False)
    original_inputs = pd.read_json(root / "input_scalar_probes.json")
    fresh_inputs = pd.read_json(root / "fresh_input_scalar_probes.json")
    pairs = original_inputs.merge(fresh_inputs, on=["dataset", "probe"], suffixes=("_original", "_fresh"), validate="one_to_one")
    if len(pairs) != 90 or (pairs.episode_fingerprint_original == pairs.episode_fingerprint_fresh).any():
        raise ValueError("fresh input-probe protocol drift")
    pairs[["dataset", "probe", "roc_auc_original", "roc_auc_fresh"]].to_csv(root / "input_probe_replication.csv", index=False)
    (root / "fresh_validation.json").write_text(json.dumps({"replay_cells": len(fresh), "input_probe_pairs": len(pairs),
        "same_model_weights": True, "different_episode_fingerprints": True, "training_seeds": [0],
        "eval_episode_offsets": [0, 100003], "independent_domain_replication": False}, indent=2) + "\n")

    coverage = pd.read_json(root / "source_coverage_cells.json")
    full = baseline[(baseline.decoder == "full_model") & (baseline.model_id != "random_init")].copy()
    full["source"] = full.model_id.str.removeprefix("ss_")
    if len(coverage) != 180 or coverage.duplicated(["dataset", "source", "policy", "reference_mode"]).any():
        raise ValueError("incomplete coverage matrix")
    combined = coverage[~coverage.source_is_target].merge(full[["dataset", "source", "roc_auc", "episode_fingerprint"]],
        on=["dataset", "source"], suffixes=("", "_replay"), validate="many_to_one")
    if not (combined.episode_fingerprint == combined.episode_fingerprint_replay).all():
        raise ValueError("coverage and replay refer to different queries")
    rows = []
    for (target, policy, mode), group in combined.groupby(["dataset", "policy", "reference_mode"]):
        if len(group) != 8:
            raise ValueError("expected eight foreign donors")
        rows.append({"dataset": target, "policy": policy, "reference_mode": mode,
                     "foreign_donors": 8, "spearman_source_mean_similarity_vs_auc": group.mean_nearest_cosine_nonzero.corr(group.roc_auc, method="spearman")})
    pd.DataFrame(rows).to_csv(root / "source_coverage_cell_correlations.csv", index=False)
    print(pairs[pairs.probe.isin(["scalar_context_coherence", "scalar_center_indegree"])][["dataset", "probe", "roc_auc_original", "roc_auc_fresh"]].to_string(index=False))
    print(json.dumps({"fresh_replay_cells": len(fresh), "coverage_cells": len(coverage),
                      "query_associations_are_descriptive": True}, indent=2))


if __name__ == "__main__":
    main()
