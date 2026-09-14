#!/usr/bin/env python3
"""Isolate zero-self scoring on the frozen AA-DC/MLP validation rescue curve."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def evaluate(pos, neg, baseline_hits):
    cutoff = float(np.partition(neg, -50)[-50])
    hits = pos > cutoff
    return {"hits_at_50": float(hits.mean()), "negative50": cutoff,
            "recovered_positives": int((hits & ~baseline_hits).sum()),
            "lost_positives": int((~hits & baseline_hits).sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    data = Path(__file__).parent / "data"
    receipt = json.loads((data / "validation_receipt.json").read_text())
    prior = json.loads((data / "results.json").read_text())
    digest = hashlib.sha256(args.scores.read_bytes()).hexdigest()
    assert digest == receipt["score_archive_sha256"]
    with np.load(args.scores) as archive:
        scores = {key: archive[key] for key in archive.files}
    pos_self = scores["positive_edges"][:, 0] == scores["positive_edges"][:, 1]
    neg_self = scores["negative_edges"][:, 0] == scores["negative_edges"][:, 1]
    ap, an = scores["aadc_positive"].astype(float), scores["aadc_negative"].astype(float)
    at = float(np.partition(an, -50)[-50])
    ah = ap > at
    assert not ap[pos_self].any() and not an[neg_self].any()
    rows = []
    for seed, original in enumerate(prior["seeds"]):
        assert original["seed"] == seed
        mp, mn = scores[f"mlp{seed}_positive"].astype(float), scores[f"mlp{seed}_negative"].astype(float)
        mt = float(np.partition(mn, -50)[-50])
        ep, en = np.exp(mp - mt), np.exp(mn - mt)
        standalone_before = evaluate(ep, en, ah)
        assert standalone_before["hits_at_50"] == original["mlp_hits_at_50"]
        ep_selfzero, en_selfzero = ep.copy(), en.copy()
        ep_selfzero[pos_self], en_selfzero[neg_self] = 0., 0.
        curve = []
        for previous in original["rescue_curve"]:
            alpha = previous["alpha"]
            cp, cn = np.maximum(ap / at, alpha * ep), np.maximum(an / at, alpha * en)
            baseline = evaluate(cp, cn, ah)
            for key in ("hits_at_50", "recovered_positives", "lost_positives"):
                assert baseline[key] == previous[key]
            xp, xn = cp.copy(), cn.copy()
            xp[pos_self], xn[neg_self] = 0., 0.
            assert np.array_equal(xp[~pos_self], cp[~pos_self])
            assert np.array_equal(xn[~neg_self], cn[~neg_self])
            changed = evaluate(xp, xn, ah)
            row = {"alpha": alpha, "original": baseline, "zero_self": changed,
                   "delta_pp_vs_original_combination": 100 * (changed["hits_at_50"] - baseline["hits_at_50"]),
                   "delta_pp_vs_aadc": 100 * (changed["hits_at_50"] - ah.mean()),
                   "remaining_new_negatives_above_original_aadc_threshold": int(((xn > 1) & ~(an > at)).sum())}
            curve.append(row)
        rows.append({"seed": seed, "standalone_mlp_original": standalone_before,
                     "standalone_mlp_zero_self": evaluate(ep_selfzero, en_selfzero, ah), "curve": curve})
    aggregate = []
    for i, cell in enumerate(rows[0]["curve"]):
        aggregate.append({"alpha": cell["alpha"],
                          "original_mean_hits_at_50": float(np.mean([s["curve"][i]["original"]["hits_at_50"] for s in rows])),
                          "zero_self_mean_hits_at_50": float(np.mean([s["curve"][i]["zero_self"]["hits_at_50"] for s in rows])),
                          "zero_self_mean_delta_pp_vs_aadc": float(np.mean([s["curve"][i]["delta_pp_vs_aadc"] for s in rows]))})
    result = {"complete": True, "classification": "post_hoc_validation_score_intervention",
              "test_scored": False, "source_archive_sha256": digest,
              "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "positive_count": len(pos_self), "negative_count": len(neg_self),
              "positive_self_pairs": int(pos_self.sum()), "negative_self_pairs": int(neg_self.sum()),
              "aadc_hits_at_50": float(ah.mean()), "original_curve_cells_verified": 24,
              "corrected_curve_cells": 24, "all_non_self_scores_unchanged": True,
              "normalization": "original model thresholds held fixed; final combined cutoff recomputed",
              "qualification": "Existing validation panel used for model selection and this post-hoc diagnostic; no independent holdout claim.",
              "seeds": rows, "aggregate": aggregate}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"aggregate": aggregate, "standalone": [
        {"seed": s["seed"], "before": s["standalone_mlp_original"]["hits_at_50"],
         "after": s["standalone_mlp_zero_self"]["hits_at_50"]} for s in rows]}, indent=2))


if __name__ == "__main__":
    main()
