#!/usr/bin/env python3
"""Create neutral, disaggregated final-core result figures.

This suite uses raw architecture-native metrics and never aggregates across
training seeds or graph orders. Run locally with:

    /opt/homebrew/bin/python3.11 \
      scripts/experiments/analysis/transfer/matrices/native_objective/final_core/plot_neutral_detailed.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import numpy as np


HERE = Path(__file__).resolve().parent
DATA = HERE / "data/results_full_long.tsv"
OUT = HERE / "figures/neutral_detailed"

GRAPHS = (
    "ukr_rus",
    "covid",
    "midterm",
    "covid_political",
    "election2020",
    "ukr_rus_suspended",
    "twibot20",
    "cp_hk",
    "facebook_page_reference",
)
SHORT = {
    "ukr_rus": "UKR/RUS",
    "covid": "COVID-19",
    "midterm": "Midterm",
    "covid_political": "COVID-pol.",
    "election2020": "Election ’20",
    "ukr_rus_suspended": "UKR/RUS susp.",
    "twibot20": "TwiBot-20",
    "cp_hk": "CP-HK",
    "facebook_page_reference": "Facebook",
}
ORDERS = ("A", "B", "C")
BLUE = "#3b78a7"
INK = "#202020"
MUTED = "#737373"
GRID = "#dddddd"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "pdf.fonttype": 42,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def load_rows() -> list[dict[str, str]]:
    with DATA.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 1944 or len({row["cell_id"] for row in rows}) != 1944:
        raise ValueError("canonical table is not the exact 1,944-cell design")
    return rows


def observed(
    rows: list[dict[str, str]],
    *,
    architecture: str,
    component: str,
    seed_slot: int,
    order: str = "",
) -> list[dict[str, str]]:
    result = [
        row
        for row in rows
        if row["architecture"] == architecture
        and row["component"] == component
        and row["result_status"] == "observed"
        and int(row["training_seed_slot"]) == seed_slot
        and (not order or row["order"] == order)
    ]
    expected = 81
    if len(result) != expected:
        raise ValueError(
            f"{architecture} {component} seed-slot {seed_slot} order {order or '-'} "
            f"has {len(result)} rows, expected {expected}"
        )
    return result


def clean_axis(ax: Any) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#bdbdbd")
    ax.spines["bottom"].set_color("#bdbdbd")
    ax.grid(color=GRID, lw=0.55)
    ax.set_axisbelow(True)


def save(fig: Any, relative_stem: str) -> tuple[str, str]:
    png = OUT / f"{relative_stem}.png"
    pdf = OUT / f"{relative_stem}.pdf"
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=230, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png.relative_to(OUT).as_posix(), pdf.relative_to(OUT).as_posix()


def text_color(cmap: Any, norm: Any, value: float) -> str:
    red, green, blue, _ = cmap(norm(value))
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "white" if luminance < 0.48 else INK


def matrix_array(rows: list[dict[str, str]]) -> np.ndarray:
    values: dict[tuple[str, str], float] = {}
    for row in rows:
        train_graphs = json.loads(row["train_graphs"])
        if len(train_graphs) != 1:
            raise ValueError("matrix row does not have exactly one training graph")
        key = (train_graphs[0], row["test_graph"])
        if key in values:
            raise ValueError(f"duplicate matrix key {key}")
        values[key] = float(row["primary_value"])
    expected = {(source, target) for source in GRAPHS for target in GRAPHS}
    if set(values) != expected:
        raise ValueError("matrix rows are not an exact 9 x 9 grid")
    return np.array([[values[(source, target)] for target in GRAPHS] for source in GRAPHS])


def plot_matrix(
    rows: list[dict[str, str]],
    *,
    architecture: str,
    seed_label: str,
    relative_stem: str,
    norm: Any,
    cmap_name: str,
    metric_label: str,
    value_format: str,
) -> tuple[str, str]:
    values = matrix_array(rows)
    cmap = plt.get_cmap(cmap_name)
    fig, ax = plt.subplots(figsize=(8.1, 7.0), constrained_layout=True)
    image = ax.imshow(values, cmap=cmap, norm=norm, aspect="equal")
    for i in range(len(GRAPHS)):
        for j in range(len(GRAPHS)):
            ax.text(
                j,
                i,
                format(values[i, j], value_format),
                ha="center",
                va="center",
                fontsize=6.3,
                color=text_color(cmap, norm, float(values[i, j])),
            )
    ax.set_xticks(range(len(GRAPHS)), [SHORT[graph] for graph in GRAPHS], rotation=42, ha="right")
    ax.set_yticks(range(len(GRAPHS)), [SHORT[graph] for graph in GRAPHS])
    ax.set_xlabel("evaluation target")
    ax.set_ylabel("training source")
    ax.set_title(f"{architecture} matrix · training seed {seed_label}", loc="left", pad=12)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    colorbar.set_label(metric_label)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return save(fig, relative_stem)


def order_sequence(rows: list[dict[str, str]]) -> tuple[str, ...]:
    by_rung: dict[int, str] = {}
    for row in rows:
        rung = int(row["rung"])
        added = row["added_graph"]
        if rung in by_rung and by_rung[rung] != added:
            raise ValueError(f"rung {rung} has conflicting added graphs")
        by_rung[rung] = added
    if set(by_rung) != set(range(1, 10)):
        raise ValueError("ladder does not have exactly rungs 1 through 9")
    return tuple(by_rung[rung] for rung in range(1, 10))


def ladder_values(rows: list[dict[str, str]]) -> dict[str, list[float]]:
    values: dict[tuple[str, int], float] = {}
    for row in rows:
        key = (row["test_graph"], int(row["rung"]))
        if key in values:
            raise ValueError(f"duplicate ladder key {key}")
        values[key] = float(row["primary_value"])
    expected = {(target, rung) for target in GRAPHS for rung in range(1, 10)}
    if set(values) != expected:
        raise ValueError("ladder rows are not an exact 9 targets x 9 rungs grid")
    return {target: [values[(target, rung)] for rung in range(1, 10)] for target in GRAPHS}


def plot_ladder(
    rows: list[dict[str, str]],
    *,
    architecture: str,
    seed_label: str,
    order: str,
    relative_stem: str,
    metric_label: str,
    y_limits: tuple[float, float],
    log_scale: bool,
) -> tuple[str, str]:
    values = ladder_values(rows)
    sequence = order_sequence(rows)
    fig, axes = plt.subplots(3, 3, figsize=(12.0, 8.5), sharex=True, sharey=True)
    rungs = np.arange(1, 10)
    for index, target in enumerate(GRAPHS):
        ax = axes.flat[index]
        ax.plot(rungs, values[target], color=BLUE, lw=1.45, marker="o", ms=3.2)
        ax.set_title(SHORT[target], loc="left")
        ax.set_xlim(0.7, 9.3)
        ax.set_ylim(*y_limits)
        if log_scale:
            ax.set_yscale("log")
        ax.set_xticks(rungs)
        if index % 3 == 0:
            ax.set_ylabel(metric_label)
        if index >= 6:
            ax.set_xlabel("rung")
        clean_axis(ax)
    figure_title = f"{architecture} ladder · training seed {seed_label} · order {order}"
    fig.suptitle(figure_title, x=0.07, ha="left", fontsize=13)
    additions = "  ·  ".join(f"{rung} {SHORT[graph]}" for rung, graph in enumerate(sequence, 1))
    fig.text(0.5, 0.018, "graph added by rung:  " + additions, ha="center", va="bottom", color=MUTED, fontsize=7.8)
    fig.tight_layout(rect=(0.03, 0.065, 1.0, 0.96), h_pad=1.2, w_pad=1.2)
    return save(fig, relative_stem)


def global_range(rows: list[dict[str, str]], architecture: str, component: str) -> tuple[float, float]:
    values = [
        float(row["primary_value"])
        for row in rows
        if row["architecture"] == architecture
        and row["component"] == component
        and row["result_status"] == "observed"
    ]
    lower, upper = min(values), max(values)
    if architecture == "SAMGPT":
        return lower * 0.75, upper * 1.35
    padding = (upper - lower) * 0.05
    return max(0.0, lower - padding), upper + padding


def write_index(records: list[dict[str, str]]) -> None:
    lines = [
        "# Neutral figure index",
        "",
        "| section | architecture | training seed | order | metric | source rows | PNG | PDF |",
        "|---|---|---:|:---:|---|---:|---|---|",
    ]
    for record in records:
        lines.append(
            "| {section} | {architecture} | {training_seed} | {order} | "
            "{metric} | {source_rows} | [{png}]({png}) | [{pdf}]({pdf}) |".format(
                **record
            )
        )
    (OUT / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = load_rows()
    records: list[dict[str, str]] = []

    prodigy_matrix_range = global_range(rows, "PRODIGY", "matrix")
    for seed_slot in (0, 1, 2):
        selected = observed(rows, architecture="PRODIGY", component="matrix", seed_slot=seed_slot)
        stem = f"matrix/prodigy_seed_{seed_slot}_matrix"
        png, pdf = plot_matrix(
            selected,
            architecture="PRODIGY",
            seed_label=str(seed_slot),
            relative_stem=stem,
            norm=Normalize(*prodigy_matrix_range),
            cmap_name="viridis",
            metric_label="neighbor-matching accuracy",
            value_format=".3f",
        )
        records.append({"section": "matrix", "architecture": "PRODIGY", "training_seed": str(seed_slot), "order": "", "metric": "neighbor_matching_accuracy", "source_rows": "81", "png": png, "pdf": pdf})

    samgpt_matrix = observed(rows, architecture="SAMGPT", component="matrix", seed_slot=0)
    samgpt_matrix_range = global_range(rows, "SAMGPT", "matrix")
    png, pdf = plot_matrix(
        samgpt_matrix,
        architecture="SAMGPT",
        seed_label="39",
        relative_stem="matrix/samgpt_seed_39_matrix",
        norm=LogNorm(*samgpt_matrix_range),
        cmap_name="viridis",
        metric_label="GraphCL BCE loss (log color scale)",
        value_format=".1e",
    )
    records.append({"section": "matrix", "architecture": "SAMGPT", "training_seed": "39", "order": "", "metric": "graphcl_bce_loss", "source_rows": "81", "png": png, "pdf": pdf})

    prodigy_ladder_range = global_range(rows, "PRODIGY", "ladder")
    for seed_slot in (0, 1, 2):
        for order in ORDERS:
            selected = observed(rows, architecture="PRODIGY", component="ladder", seed_slot=seed_slot, order=order)
            stem = f"ladder/prodigy_seed_{seed_slot}_order_{order}"
            png, pdf = plot_ladder(
                selected,
                architecture="PRODIGY",
                seed_label=str(seed_slot),
                order=order,
                relative_stem=stem,
                metric_label="NM accuracy",
                y_limits=prodigy_ladder_range,
                log_scale=False,
            )
            records.append({"section": "ladder", "architecture": "PRODIGY", "training_seed": str(seed_slot), "order": order, "metric": "neighbor_matching_accuracy", "source_rows": "81", "png": png, "pdf": pdf})

    samgpt_ladder_range = global_range(rows, "SAMGPT", "ladder")
    for order in ORDERS:
        selected = observed(rows, architecture="SAMGPT", component="ladder", seed_slot=0, order=order)
        stem = f"ladder/samgpt_seed_39_order_{order}"
        png, pdf = plot_ladder(
            selected,
            architecture="SAMGPT",
            seed_label="39",
            order=order,
            relative_stem=stem,
            metric_label="GraphCL BCE loss",
            y_limits=samgpt_ladder_range,
            log_scale=True,
        )
        records.append({"section": "ladder", "architecture": "SAMGPT", "training_seed": "39", "order": order, "metric": "graphcl_bce_loss", "source_rows": "81", "png": png, "pdf": pdf})

    if len(records) != 16:
        raise ValueError(f"expected 16 neutral figure specifications, found {len(records)}")
    write_index(records)
    print(f"FINAL_CORE_NEUTRAL_FIGURES_OK figures={len(records)} formats=png,pdf output={OUT}")


if __name__ == "__main__":
    main()
