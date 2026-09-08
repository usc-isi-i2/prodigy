#!/usr/bin/env python3
"""Build the seed-0 paired PinSAGE versus final-core GraphSAGE table."""
from __future__ import annotations

import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]
PIN_ROOT = HERE / "data/fixed_test_seed0/results/seed_0"
SAGE_ROOT = (
    REPO / "scripts/experiments/analysis/transfer/matrices/cross_model/final_core/data/"
    "prodigy_final_core/fixed_test/results/seed_0"
)
PAIRS = {
    "pinsage_covid": "ss_covid",
    "pinsage_election": "ss_election2020",
    "pinsage_loo_twibot20": "ordB_r8",
    "pinsage_twibot20": "ss_twibot20",
}
TARGETS = ("covid_political", "election2020", "ukr_rus_suspended", "twibot20")


rows = []
for pinsage_model, sage_model in PAIRS.items():
    for target in TARGETS:
        pin = json.loads((PIN_ROOT / pinsage_model / f"{target}.json").read_text())
        sage = json.loads((SAGE_ROOT / sage_model / f"{target}.json").read_text())
        for key in ("episode_plan_fingerprint", "observed_episode_fingerprint"):
            if pin[key] != sage[key]:
                raise ValueError(f"unpaired stream for {pinsage_model}/{target}: {key}")
        rows.append({
            "pinsage_model": pinsage_model, "graphsage_model": sage_model,
            "target": target, "pinsage_accuracy": pin["accuracy"],
            "graphsage_accuracy": sage["score"],
            "accuracy_delta": pin["accuracy"] - sage["score"],
            "pinsage_f1_macro": pin["f1_macro"],
            "pinsage_roc_auc_ovr_macro": pin["roc_auc_ovr_macro"],
            "episode_plan_fingerprint": pin["episode_plan_fingerprint"],
            "observed_episode_fingerprint": pin["observed_episode_fingerprint"],
        })

output = HERE / "data/fixed_test_seed0/comparison.csv"
with output.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(output)
