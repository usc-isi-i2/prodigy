"""Post-hoc failure audit for compact-joint assessment scores; never reads test."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


FEATURES = [
    "log_aa", "log_l3", "external_ratio", "lcc_min", "lcc_max",
    "log_degree_min", "log_degree_max", "raw_cosine", "log_pair_events",
    "last_event_age", "log_recent_min", "log_recent_max", "aa_zero",
]
ALPHAS = (0.0, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cutoff(scores: np.ndarray) -> float:
    return float(np.partition(scores, -50)[-50])


def hitmask(positive: np.ndarray, negative: np.ndarray) -> np.ndarray:
    return positive > cutoff(negative)


def top50(scores: np.ndarray) -> np.ndarray:
    result = np.zeros(len(scores), dtype=bool)
    result[np.argsort(-scores, kind="stable")[:50]] = True
    return result


def profile(features: np.ndarray, mask: np.ndarray) -> dict[str, object]:
    selected = features[mask]
    return {
        "count": int(mask.sum()),
        "aa_zero_fraction": float(selected[:, 12].mean()),
        "l3_nonzero_fraction": float((selected[:, 1] > 0).mean()),
        "both_recently_active_fraction": float((selected[:, 10] > 0).mean()),
        "cosine_at_least_0_96_fraction": float((selected[:, 7] >= 0.96).mean()),
        "min_degree_at_least_10_fraction": float(
            (selected[:, 5] >= np.log1p(10)).mean()
        ),
        "repeat_pair_fraction": float((selected[:, 8] > 0).mean()),
        "medians": {
            FEATURES[i]: float(np.median(selected[:, i]))
            for i in (0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    results_path = args.evidence / "results.json"
    results = json.loads(results_path.read_text())
    assert results["complete"] and results["test_scored"] is False
    panel_path, scores_path = args.runtime / "year2018.npz", args.runtime / "scores.npz"
    assert sha256(panel_path) == results["panel_sha256"]
    assert sha256(scores_path) == results["score_sha256"]
    panel, scores = np.load(panel_path), np.load(scores_path)
    assert len(panel["pos"]) == 60084 and len(panel["neg"]) == 100000

    baseline_hits = hitmask(panel["official_bp"], panel["official_bn"])
    baseline_top = top50(panel["official_bn"])
    joint_hits = np.stack([
        hitmask(scores[f"joint_seed{seed}_p"], scores[f"joint_seed{seed}_n"])
        for seed in range(3)
    ])
    joint_top = np.stack([top50(scores[f"joint_seed{seed}_n"]) for seed in range(3)])

    positive_masks = {
        "aadc_miss": ~baseline_hits,
        "joint_recovered_all_seeds": (~baseline_hits) & joint_hits.all(axis=0),
        "joint_recovered_any_seed": (~baseline_hits) & joint_hits.any(axis=0),
        "joint_lost_all_seeds": baseline_hits & (~joint_hits).all(axis=0),
        "persistent_miss_all_seeds": (~baseline_hits) & (~joint_hits).all(axis=0),
    }
    negative_masks = {
        "joint_promoted_all_seeds": (~baseline_top) & joint_top.all(axis=0),
        "joint_promoted_any_seed": (~baseline_top) & joint_top.any(axis=0),
    }

    fusion = []
    base_cutoff = cutoff(panel["official_bn"])
    base_positive = panel["official_bp"] / base_cutoff
    base_negative = panel["official_bn"] / base_cutoff
    for alpha in ALPHAS:
        values = []
        for seed in range(3):
            joint_positive = scores[f"joint_seed{seed}_p"]
            joint_negative = scores[f"joint_seed{seed}_n"]
            joint_cutoff = cutoff(joint_negative)
            if alpha == 0:
                combined_positive, combined_negative = base_positive, base_negative
            else:
                combined_positive = np.maximum(
                    base_positive,
                    alpha * np.exp(np.clip(joint_positive - joint_cutoff, -30, 30)),
                )
                combined_negative = np.maximum(
                    base_negative,
                    alpha * np.exp(np.clip(joint_negative - joint_cutoff, -30, 30)),
                )
            values.append(float(hitmask(combined_positive, combined_negative).mean()))
        fusion.append({"alpha": alpha, "seed_hits_at_50": values, "mean": float(np.mean(values))})

    best = max(fusion, key=lambda row: row["mean"])
    output = {
        "classification": "post_hoc_hypothesis_generating_validation_diagnostic",
        "complete": True,
        "test_read_or_scored": False,
        "source_revision": results["revision"],
        "source_results_sha256": sha256(results_path),
        "raw_panel_and_score_hashes_verified": True,
        "positive_profiles": {
            name: profile(panel["pfeatures"], mask) for name, mask in positive_masks.items()
        },
        "negative_profiles": {
            name: profile(panel["nfeatures"], mask) for name, mask in negative_masks.items()
        },
        "joint_oracle_union_hit_counts_with_aadc": [
            int((baseline_hits | joint_hits[seed]).sum()) for seed in range(3)
        ],
        "fusion_definition": (
            "max(AA_DC/AA_DC_negative50, alpha*exp(clip(joint_logit-joint_negative50,-30,30))); "
            "each combined negative cutoff is recomputed"
        ),
        "fusion_curve": fusion,
        "best_inspected_common_alpha": best["alpha"],
        "best_inspected_mean_hits_at_50": best["mean"],
        "qualification": (
            "Alpha and hypotheses were inspected on repeatedly used official 2018 validation. "
            "This is not an independently selected model or expected test performance."
        ),
    }
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
