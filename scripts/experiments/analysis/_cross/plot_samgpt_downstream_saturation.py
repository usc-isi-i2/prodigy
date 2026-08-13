#!/usr/bin/env python3
"""Plot SAMGPT downstream ROC-AUC across pretraining updates."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
ANALYSIS_ROOT = HERE.parent
FIGURE_ROOT = HERE / "figures"

SERIES = (
    (
        "COVID only",
        ANALYSIS_ROOT / "samgpt_covid_saturation" / "data" / "validation_trajectory.csv",
        "#2a78d6",
        "o",
    ),
    (
        "Five-source mixture",
        ANALYSIS_ROOT / "samgpt_c5_saturation" / "data" / "validation_trajectory.csv",
        "#d85a30",
        "s",
    ),
)

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "pdf.fonttype": 42,
    }
)


def load(path: Path) -> tuple[list[int], list[float]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return (
        [int(row["epoch"]) for row in rows],
        [float(row["roc_auc_mean"]) for row in rows],
    )


def main() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.35))

    for label, path, color, marker in SERIES:
        updates, auc = load(path)
        ax.plot(
            updates,
            auc,
            color=color,
            marker=marker,
            markersize=5.5,
            linewidth=2.0,
            label=label,
            zorder=3,
        )

    ax.axvspan(0, 1000, color="#8f8d87", alpha=0.08, linewidth=0)
    ax.text(
        510,
        0.674,
        "early-update region",
        ha="center",
        va="bottom",
        color="#77756f",
        fontsize=8.5,
    )

    ax.set_title(
        "SAMGPT downstream performance saturates early",
        loc="left",
        fontsize=14,
        fontweight="medium",
        pad=12,
    )
    ax.text(
        0,
        1.01,
        "TwiBot-20 validation ROC-AUC · 500 fixed episodes · one training seed",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="#77756f",
    )
    ax.set_xlabel("pretraining updates")
    ax.set_ylabel("validation ROC-AUC")
    ax.set_xlim(-80, 4080)
    ax.set_ylim(0.668, 0.716)
    ax.set_xticks([0, 500, 1000, 2000, 3000, 4000])
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c3c2b7")
    ax.tick_params(colors="#77756f")
    ax.legend(frameon=False, loc="lower right")

    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        output = FIGURE_ROOT / f"samgpt_downstream_saturation.{suffix}"
        metadata = {"CreationDate": None, "ModDate": None} if suffix == "pdf" else None
        fig.savefig(output, dpi=220, bbox_inches="tight", metadata=metadata)
        print(f"wrote {output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
