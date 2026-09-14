#!/usr/bin/env python3
"""Audit the saved validation archive and identify low-weight negative promotion."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).parent / "data"
    receipt = json.loads((root / "validation_receipt.json").read_text())
    result = json.loads((root / "results.json").read_text())
    digest = hashlib.sha256(args.scores.read_bytes()).hexdigest()
    assert digest == receipt["score_archive_sha256"]
    z = np.load(args.scores)
    pos, neg = z["positive_edges"], z["negative_edges"]
    assert pos.shape == (60084, 2) and neg.shape == (100000, 2)
    ap, an = z["aadc_positive"], z["aadc_negative"]
    at = float(np.sort(an)[-50])
    ah = ap > at
    rows = []
    for seed in range(3):
        mp, mn = z[f"mlp{seed}_positive"], z[f"mlp{seed}_negative"]
        mt = float(np.sort(mn)[-50])
        saved = result["seeds"][seed]
        assert int(((mp > mt) & ~ah).sum()) == saved["mlp_recovers_aadc_misses"]
        for row in saved["rescue_curve"]:
            cp = np.maximum(ap.astype(float) / at, row["alpha"] * np.exp(mp.astype(float) - mt))
            cn = np.maximum(an.astype(float) / at, row["alpha"] * np.exp(mn.astype(float) - mt))
            hits = cp > np.sort(cn)[-50]
            assert float(hits.mean()) == row["hits_at_50"]
            assert int((hits & ~ah).sum()) == row["recovered_positives"]
            assert int((~hits & ah).sum()) == row["lost_positives"]
        promoted = np.flatnonzero((.1 * np.exp(mn.astype(float) - mt) > 1) & (an <= at))
        rows.append({"seed": seed, "new_negative_indices_at_alpha_0_1": promoted.tolist(),
                     "are_self_pairs": (neg[promoted, 0] == neg[promoted, 1]).tolist(),
                     "aadc_scores": an[promoted].tolist(), "mlp_logits": mn[promoted].tolist()})
    print(json.dumps({"archive_sha256_verified": digest, "all_24_curve_cells_recomputed": True,
                      "positive_self_pairs": int((pos[:, 0] == pos[:, 1]).sum()),
                      "negative_self_pairs": int((neg[:, 0] == neg[:, 1]).sum()),
                      "test_scored": False, "low_weight_negative_audit": rows}, indent=2))


if __name__ == "__main__":
    main()
