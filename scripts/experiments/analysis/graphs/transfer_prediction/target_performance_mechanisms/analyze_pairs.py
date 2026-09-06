"""Stage-resolved Ukraine/TwiBot substitution contrasts at fixed pair partners."""
import json
from pathlib import Path

import pandas as pd


def main():
    root = Path(__file__).parent / "data"
    cells = pd.DataFrame([json.loads(line) for path in sorted((root / "pair_replay").glob("*.jsonl"))
                          for line in path.read_text().splitlines()])
    if len(cells) != 476 or cells.duplicated(["dataset", "model_id", "decoder"]).any():
        raise ValueError("expected exactly 14 paired models x 2 targets x 17 decoders")
    full = cells[cells.decoder == "full_model"]
    if len(full) != 28 or full.official_metric_max_abs_error.isna().any() or (full.official_metric_max_abs_error > 1e-5).any():
        raise ValueError("unverified reference parity")
    if (full.official_decision_metric_max_abs_error > 1e-6).any():
        raise ValueError("decision metrics drifted")
    rows = []
    for target, part in cells.groupby("dataset"):
        if part.episode_fingerprint.nunique() != 1 or set(part.episodes) != {128}:
            raise ValueError("episode mismatch")
        part = part.copy()
        part["donor"] = part.sources.map(lambda sources: "ukr_rus" if "ukr_rus" in sources else "twibot20")
        part["partner"] = part.sources.map(lambda sources: next(x for x in sources if x not in {"ukr_rus", "twibot20"}))
        part = part[part.partner != target]  # six matched foreign partners
        for (partner, decoder), match in part.groupby(["partner", "decoder"]):
            if len(match) != 2 or set(match.donor) != {"ukr_rus", "twibot20"}:
                raise ValueError("unmatched donor substitution")
            values = match.set_index("donor").roc_auc
            rows.append({"dataset": target, "partner": partner, "decoder": decoder,
                         "ukraine_auc": values.ukr_rus, "twibot_auc": values.twibot20,
                         "ukraine_minus_twibot_auc": values.ukr_rus - values.twibot20})
    pairs = pd.DataFrame(rows)
    pairs.to_csv(root / "pair_stage_contrasts.csv", index=False)
    summary = pairs.groupby(["dataset", "decoder"]).ukraine_minus_twibot_auc.agg(
        mean="mean", minimum="min", maximum="max", positive=lambda x: int((x > 0).sum()), matched_partners="size")
    summary.to_csv(root / "pair_stage_summary.csv")
    print(summary.loc[(slice(None), ["raw_center/ridge", "S0_conv_center/ridge", "S0_pool/ridge", "U1_pre_meta/ridge", "M2_post_meta/ridge", "final_input/ridge", "full_model"]), :].to_string())
    receipt = {"rows": len(cells), "official_parity_cells": len(full),
               "max_official_metric_error": float(full.official_metric_max_abs_error.max()),
               "training_seeds": [0], "foreign_partners_per_target": 6,
               "independent_training_seed_replications": False}
    (root / "pair_validation.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
