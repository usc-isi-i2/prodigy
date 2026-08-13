#!/usr/bin/env python3
"""Per-rung AUC delta plot for the NM interpolation ladder.

For each test graph, plot the change in AUC when moving from each ladder rung to the
next (Δ vs previous rung). Each graph's line spikes at the rung where THAT graph enters
the training merge and hovers near zero otherwise — i.e. the marginal contribution of
each added source, resolved per test graph.

x = L1, L2, ..., all8 (ladder rungs; L1 has no previous rung → 0).
y = ΔAUC vs the previous rung. One colored line per test graph.

Reads data/nm_ladder_full.csv if present; else uses the embedded matched-40k
values. Writes nm_ladder_per_step_delta.{pdf,png} next to this script.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CANON = ["ukr_rus_twitter", "covid19_twitter", "midterm", "covid_political",
         "election2020", "ukr_rus_suspended", "twibot20", "cp_hk_twitter"]
SHORT = {"ukr_rus_twitter": "ukr", "covid19_twitter": "covid", "midterm": "midterm",
         "covid_political": "cov_pol", "election2020": "elec20",
         "ukr_rus_suspended": "ukr_susp", "twibot20": "twibot20", "cp_hk_twitter": "cp_hk"}
RUNG_LABEL = ["L1", "L2", "L3", "L4", "L5", "L6", "L7", "all8"]
COLORS = ["#2a78d6", "#1baf7a", "#eda100", "#008300",
          "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]
MARKERS = ["o", "^", "s", "D", "*", "X", "P", "v"]

# Embedded matched-40k ladder AUCs, rungs 1..8 (fallback if the CSV is absent).
LADDER = [
    [.9480, .9730, .8740, .8490, .8280, .7710, .9210, .7240],
    [.9450, .9800, .8850, .8430, .8280, .7750, .9250, .7260],
    [.9410, .9780, .9150, .8300, .8150, .7770, .9270, .7200],
    [.9344, .9753, .9093, .9113, .8297, .7768, .9234, .7235],
    [.9346, .9754, .9086, .9102, .9259, .7693, .9254, .7261],
    [.9325, .9744, .9073, .9106, .9241, .9340, .9242, .7239],
    [.9321, .9748, .9033, .9076, .9198, .9256, .9377, .7267],
    [.9340, .9750, .9080, .9060, .9200, .9310, .9370, .8670],
]


def load_ladder() -> list[list[float]]:
    here = Path(__file__).resolve().parent
    csv_path = here / "data" / "nm_ladder_full.csv"
    if csv_path.is_file():
        try:
            rows: dict[int, list[float]] = {}
            with csv_path.open(encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    if not r.get("rung"):
                        continue
                    rows[int(r["rung"])] = [float(r[c]) for c in CANON]
            if all(k in rows for k in range(1, 9)):
                return [rows[k] for k in range(1, 9)]
        except (ValueError, KeyError):
            pass
    return LADDER


def main() -> None:
    L = load_ladder()
    # deltas[k] = L[k] - L[k-1] elementwise; rung 0 (L1) = 0 (no previous rung).
    deltas = [[0.0] * 8]
    for k in range(1, 8):
        deltas.append([L[k][j] - L[k - 1][j] for j in range(8)])
    xs = list(range(8))

    fig, ax = plt.subplots(figsize=(8.4, 4.7))
    ax.axhline(0, color="#888780", lw=1.0, zorder=1)
    for j, key in enumerate(CANON):
        ys = [deltas[k][j] for k in range(8)]
        ax.plot(xs, ys, color=COLORS[j], marker=MARKERS[j], ms=5.5, lw=1.9,
                label=SHORT[key], zorder=3)

    ax.set_xticks(xs)
    ax.set_xticklabels(RUNG_LABEL)
    ax.set_ylabel("Δ AUC vs previous rung")
    ax.set_xlabel("ladder rung (training set grows 1 → 8 sources)")
    ax.set_title("Marginal effect of each added graph, per test graph (NM, matched-40k)")
    ax.grid(True, axis="y", color="#e1e0d9", lw=0.6, zorder=0)
    ax.legend(ncol=4, fontsize=9, frameon=False, loc="upper left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()

    out = Path(__file__).resolve().parent / "figures"
    fig.savefig(out / "nm_ladder_per_step_delta.pdf")
    fig.savefig(out / "nm_ladder_per_step_delta.png", dpi=150)
    print(f"wrote {out / 'nm_ladder_per_step_delta.pdf'} (+ .png)")


if __name__ == "__main__":
    main()
