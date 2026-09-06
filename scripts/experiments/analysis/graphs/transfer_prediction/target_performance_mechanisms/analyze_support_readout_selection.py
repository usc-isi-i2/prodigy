"""Readout-selection utility against every equally normalized fixed comparator."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PANEL = {"covid_political", "facebook_page_reference", "twibot20"}


def analyze(root):
    done = json.loads((root / "DONE.json").read_text())
    if not done["complete"] or done["smoke"] or done["metric_cells"] != 1260 or done["family_source_target_stream_cells"] != 180:
        raise ValueError("complete two-family selection test required")
    cells = pd.read_csv(root / "cells.csv")
    decisions = pd.read_csv(root / "decisions.csv")
    keys = ["stream", "target", "source", "family"]
    if len(cells) != 1260 or cells.duplicated(keys + ["condition"]).any():
        raise ValueError("missing or duplicate scores")
    if not np.isfinite(cells[["roc_auc", "accuracy", "nll"]]).all().all():
        raise ValueError("nonfinite scores")
    if len(decisions) != 57600 or decisions.duplicated(keys + ["episode", "candidate"]).any():
        raise ValueError("missing or duplicate support decisions")
    if not decisions.groupby(keys + ["episode"]).selected.sum().eq(1).all():
        raise ValueError("must select exactly one readout per episode")
    priority = {"raw": 0, "prototype": 1, "full": 2}
    expected = decisions.assign(priority=decisions.candidate.map(priority)).sort_values(
        ["support_cv_auc", "priority"], ascending=[False, True]).drop_duplicates(keys + ["episode"])
    if not expected.selected.all():
        raise ValueError("decision violates declared AUC/tie rule")
    foreign = cells[cells.source != cells.target]
    target_means = foreign.groupby(["family", "stream", "target", "condition"])[["roc_auc", "accuracy", "nll"]].mean().reset_index()
    target_means.to_csv(root / "target_means.csv", index=False)
    panel = foreign[foreign.target.isin(PANEL)]
    means = panel.groupby(["family", "stream", "condition"]).roc_auc.mean().unstack("condition")
    results = []
    for (family, stream), row in means.iterrows():
        candidates = ["raw", "prototype"] + (["full"] if family == "prodigy" else [])
        fixed = [c + "_scaled" for c in candidates]
        target_panel = target_means[(target_means.family == family) & (target_means.stream == stream) & target_means.target.isin(PANEL)]
        best_target = target_panel[target_panel.condition.isin(fixed)].groupby("target").roc_auc.max().mean()
        results.append({"family": family, "stream": stream, "selected_scaled": row.selected_scaled,
            "best_fixed_scaled": row[fixed].max(), "selected_minus_best_fixed_scaled": row.selected_scaled-row[fixed].max(),
            "beats_each_fixed_scaled": bool((row.selected_scaled > row[fixed]).all()),
            "hindsight_best_fixed_per_target": best_target,
            "selected_minus_hindsight_per_target": row.selected_scaled-best_target,
            "selected_original": row.selected_original,
            "native_original": row["full_original" if family == "prodigy" else "prototype_original"],
            **{c: row[c] for c in fixed}})
    pd.DataFrame(results).to_csv(root / "primary_panel.csv", index=False)
    all_targets = foreign.groupby(["family", "stream", "condition"])[["roc_auc", "accuracy", "nll"]].mean().reset_index()
    all_targets.to_csv(root / "all_target_means.csv", index=False)
    paired = panel.pivot(index=keys, columns="condition", values="roc_auc")
    cell_results = []
    for key, row in paired.iterrows():
        candidates = ["raw_scaled", "prototype_scaled"] + (["full_scaled"] if key[-1] == "prodigy" else [])
        cell_results.append(dict(zip(keys, key), selected_scaled=row.selected_scaled,
            best_fixed_scaled=row[candidates].max(),
            selected_minus_best_fixed_scaled=row.selected_scaled-row[candidates].max()))
    pd.DataFrame(cell_results).to_csv(root / "primary_source_target_contrasts.csv", index=False)
    proportions = decisions[decisions.source != decisions.target].groupby(["family", "stream", "target", "candidate"]).selected.mean().reset_index(name="selection_fraction")
    proportions.to_csv(root / "selection_frequencies.csv", index=False)
    write = {"complete": True, "primary_comparisons": 4,
        "primary_positive_comparisons": sum(r["beats_each_fixed_scaled"] for r in results),
        "primary_passes_each_family_each_stream": all(r["beats_each_fixed_scaled"] for r in results),
        "no_query_labels_used_in_selection": True, "additional_labeled_accounts": 0,
        "new_training_seeds": 0, "unseen_target_confirmation": False}
    (root / "summary.json").write_text(json.dumps(write, indent=2) + "\n")
    print(json.dumps(write, indent=2))
    print(pd.DataFrame(results).round(5).to_string(index=False))
    print(target_means.pivot(index=["family", "stream", "target"], columns="condition", values="roc_auc").round(4).to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--input", type=Path, default=Path(__file__).with_name("data") / "support_readout_selection")
    args = p.parse_args()
    analyze(args.input)
