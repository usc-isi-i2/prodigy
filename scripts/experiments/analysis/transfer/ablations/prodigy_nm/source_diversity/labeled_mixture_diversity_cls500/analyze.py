#!/usr/bin/env python3
"""Validate and summarize the fixed-compute labeled-mixture diversity experiment."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
TARGETS = (
    "covid_political", "election2020", "facebook_page_reference",
    "ukr_rus_suspended", "twibot20",
)


def load_jsonl(paths):
    rows = []
    for path in paths:
        rows.extend(json.loads(line) for line in Path(path).read_text().splitlines() if line.strip())
    return rows


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean(rows, metric):
    return statistics.mean(float(row[metric]) for row in rows)


def validate(heldout, controls):
    assert len(heldout) == 75, len(heldout)
    assert len(controls) == 10, len(controls)
    assert len({(r["target"], r["model_id"]) for r in heldout}) == 75
    assert len({(r["target"], r["model_id"]) for r in controls}) == 10
    for target in TARGETS:
        rows = [r for r in heldout if r["target"] == target]
        assert {k: sum(r["mixture_size"] == k for r in rows) for k in range(1, 5)} == {
            1: 4, 2: 6, 3: 4, 4: 1,
        }
        assert all(target not in r["donors"] for r in rows)
        assert len({r["episode_fingerprint"] for r in rows}) == 1
        endpoint = [r for r in controls if r["target"] == target]
        assert {r["endpoint"] for r in endpoint} == {"target_only", "all_five"}
        assert all(target in r["donors"] for r in endpoint)
        assert {r["episode_fingerprint"] for r in endpoint} == {rows[0]["episode_fingerprint"]}
    assert all(r["eval_episodes"] == 500 and r["training_steps"] == 500 for r in heldout + controls)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heldout", nargs="+", required=True)
    parser.add_argument("--controls", nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, default=HERE)
    args = parser.parse_args()
    heldout, controls = load_jsonl(args.heldout), load_jsonl(args.controls)
    for row in heldout:
        row.setdefault("endpoint", "heldout")
        row.setdefault("target_in_training", False)
    validate(heldout, controls)

    out = args.output_root
    long_rows = []
    for row in heldout + controls:
        clean = dict(row)
        clean["donors"] = ",".join(row["donors"])
        long_rows.append(clean)
    fields = [
        "target", "model_id", "endpoint", "target_in_training", "mixture_size",
        "donors", "training_steps", "training_seed", "eval_episodes",
        "episode_fingerprint", "roc_auc", "accuracy", "f1", "checkpoint",
    ]
    write_csv(out / "data" / "all_results.csv", long_rows, fields)

    summary = []
    for k in range(1, 5):
        rows = [r for r in heldout if r["mixture_size"] == k]
        summary.append({
            "scope": "macro", "target": "all", "mixture_size": k,
            "n": len(rows), "roc_auc": mean(rows, "roc_auc"),
            "accuracy": mean(rows, "accuracy"), "f1": mean(rows, "f1"),
        })
    target_means = defaultdict(dict)
    for target in TARGETS:
        for k in range(1, 5):
            rows = [r for r in heldout if r["target"] == target and r["mixture_size"] == k]
            target_means[target][k] = mean(rows, "roc_auc")
            summary.append({
                "scope": "target", "target": target, "mixture_size": k,
                "n": len(rows), "roc_auc": mean(rows, "roc_auc"),
                "accuracy": mean(rows, "accuracy"), "f1": mean(rows, "f1"),
            })
    write_csv(out / "data" / "summary.csv", summary, list(summary[0]))

    endpoint_rows = []
    for target in TARGETS:
        k4 = next(r for r in heldout if r["target"] == target and r["mixture_size"] == 4)
        target_only = next(r for r in controls if r["target"] == target and r["endpoint"] == "target_only")
        all_five = next(r for r in controls if r["target"] == target and r["endpoint"] == "all_five")
        endpoint_rows.append({
            "target": target, "heldout_k4_auc": k4["roc_auc"],
            "target_only_auc": target_only["roc_auc"], "all_five_auc": all_five["roc_auc"],
            "all_five_minus_k4": all_five["roc_auc"] - k4["roc_auc"],
            "target_only_minus_k4": target_only["roc_auc"] - k4["roc_auc"],
        })
    write_csv(out / "data" / "endpoint_controls.csv", endpoint_rows, list(endpoint_rows[0]))

    # Every target has a complete subset lattice over its four eligible donors.
    # Compare each subset with the otherwise-identical subset after adding one donor.
    by_subset = {
        (row["target"], frozenset(row["donors"])): float(row["roc_auc"])
        for row in heldout
    }
    marginal_rows = []
    for target in TARGETS:
        donors = sorted(set().union(*(
            set(row["donors"]) for row in heldout if row["target"] == target
        )))
        for donor in donors:
            deltas = []
            for (row_target, subset), value in by_subset.items():
                expanded = frozenset(set(subset) | {donor})
                if row_target == target and donor not in subset and (target, expanded) in by_subset:
                    deltas.append(by_subset[(target, expanded)] - value)
            assert len(deltas) == 7
            marginal_rows.append({
                "target": target, "added_donor": donor, "n_subset_edges": len(deltas),
                "mean_auc_delta": statistics.mean(deltas), "min_auc_delta": min(deltas),
                "max_auc_delta": max(deltas),
            })
    write_csv(out / "data" / "marginal_donor_effects.csv", marginal_rows, list(marginal_rows[0]))

    ks = np.array([1, 2, 3, 4], dtype=float)
    slopes = {
        target: float(np.polyfit(ks, [target_means[target][k] for k in range(1, 5)], 1)[0])
        for target in TARGETS
    }
    macro = [
        next(r["roc_auc"] for r in summary if r["scope"] == "macro" and r["mixture_size"] == k)
        for k in range(1, 5)
    ]
    macro_slope = float(np.polyfit(ks, macro, 1)[0])
    endpoint_means = {
        key: statistics.mean(float(row[key]) for row in endpoint_rows)
        for key in ("heldout_k4_auc", "target_only_auc", "all_five_auc",
                    "all_five_minus_k4", "target_only_minus_k4")
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for target in TARGETS:
        axes[0].plot(ks, [target_means[target][k] for k in range(1, 5)], marker="o", label=target)
    axes[0].plot(ks, macro, color="black", linewidth=3, marker="o", label="macro mean")
    axes[0].set(xlabel="Number of pretraining graphs", ylabel="Held-out ROC-AUC", xticks=ks)
    axes[0].legend(fontsize=7)
    x = np.arange(len(TARGETS))
    width = 0.25
    axes[1].bar(x - width, [r["heldout_k4_auc"] for r in endpoint_rows], width, label="4-source held out")
    axes[1].bar(x, [r["target_only_auc"] for r in endpoint_rows], width, label="target only")
    axes[1].bar(x + width, [r["all_five_auc"] for r in endpoint_rows], width, label="all five")
    axes[1].set_xticks(x, [t.replace("_", "\n") for t in TARGETS], fontsize=7)
    axes[1].set_ylabel("ROC-AUC")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    (out / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "figures" / "mixture_diversity_and_controls.png", dpi=180)
    plt.close(fig)

    lines = [
        "# Results", "", "## Answer", "",
        "**At a fixed 500-step budget, broader labeled mixtures improve held-out "
        "classification on average, but the effect is target- and donor-dependent rather "
        "than universal.**", "",
        f"Across targets, mean held-out ROC-AUC rises from {macro[0]:.4f} at one source "
        f"to {macro[-1]:.4f} at four sources (difference {macro[-1]-macro[0]:+.4f}; "
        f"linear slope {macro_slope:+.4f} per added source).", "",
        "Target-specific slopes: " + ", ".join(f"`{t}` {s:+.4f}" for t, s in slopes.items()) + ".", "",
        "The positive macro curve is driven by Covid Political and TwiBot. Election 2020 is "
        "already near ceiling, Facebook is flat on average, and Ukraine Suspended remains "
        "near chance.", "", "## Endpoint controls", "",
        f"The four-source held-out mean is {endpoint_means['heldout_k4_auc']:.4f}, versus "
        f"{endpoint_means['target_only_auc']:.4f} for target-only pretraining and "
        f"{endpoint_means['all_five_auc']:.4f} for all-five pretraining. Adding the target "
        f"to the mixture therefore changes mean AUC by {endpoint_means['all_five_minus_k4']:+.4f}; "
        "it does not produce a general in-domain jump under fixed total compute.", "",
        "The endpoint response is also heterogeneous: all-five helps Facebook strongly, is "
        "neutral on Election, and is worse than the held-out four-source model on Covid "
        "Political, Ukraine Suspended, and TwiBot.", "", "## What drives the curve", "",
        "Subset-lattice contrasts in `data/marginal_donor_effects.csv` show donor compatibility, "
        "not source count alone. TwiBot benefits most from adding Covid Political and Election; "
        "Facebook benefits from TwiBot but is hurt by Ukraine Suspended; Covid Political "
        "benefits most from Ukraine Suspended and TwiBot.", "", "## Scope", "",
        "All models use seed 0, 500 optimizer steps, and 500 paired evaluation episodes per "
        "target. Paired fingerprints remove evaluation-set variation across arms, but there is "
        "no training-seed uncertainty yet. The experiment estimates the practical effect of "
        "diversity under fixed total compute; it does not hold per-source exposure constant.",
    ]
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"validated {len(heldout)} held-out and {len(controls)} control cells")


if __name__ == "__main__":
    main()
