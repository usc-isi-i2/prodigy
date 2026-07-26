"""Mean performance per task for 1-, 2- and 3-objective models.

One panel per downstream task. Box = spread over (model, dataset) means within
an objective count; each point is one (model, dataset) cell, coloured by dataset.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
CSV = HERE / "data" / "combined_all_arms.csv"
FIGURES = HERE / "figures"

PRIMARY = {
    "classification": "roc_auc",
    "regression": "spearman",
    "static_link_prediction": "roc_auc",
}

TASK_TITLES = {
    "classification": "Classification",
    "regression": "Regression",
    "static_link_prediction": "Static link prediction",
}

METRIC_LABELS = {
    "roc_auc": "ROC-AUC",
    "spearman": "Spearman ρ",
}

K_LABELS = {1: "1 objective", 2: "2 objectives", 3: "3 objectives"}

# Marker shape encodes the objective mix, colour encodes the dataset.
MODEL_MARKERS = {
    "NM": "o",
    "CL": "^",
    "FP": "s",
    "NMCL": "D",
    "NMFP": "v",
    "CLFP": "P",
    "MIX": "*",
}

MODEL_LABELS = {
    "NM": "NM",
    "CL": "CL",
    "FP": "FP",
    "NMCL": "NM+CL",
    "NMFP": "NM+FP",
    "CLFP": "CL+FP",
    "MIX": "NM+FP+CL",
}


def load() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    df = df[df["split"].eq("test")]

    parts = []
    for task, metric in PRIMARY.items():
        sub = df[df["task"].eq(task)].copy()
        sub["score"] = sub[metric]
        sub = sub.dropna(subset=["score"])

        # Average over targets and repeated runs, then over models within k:
        # one point per (k, model, dataset).
        cell = (
            sub.groupby(["k", "model", "dataset"], as_index=False)["score"]
            .mean()
        )
        cell["task"] = task
        parts.append(cell)

    out = pd.concat(parts, ignore_index=True)
    out["k_label"] = out["k"].map(K_LABELS)
    return out


def plot(df: pd.DataFrame) -> plt.Figure:
    datasets = sorted(df["dataset"].unique())
    palette = dict(zip(datasets, sns.color_palette("tab10", len(datasets))))
    order = [K_LABELS[k] for k in (1, 2, 3)]

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))

    for ax, task in zip(axes, PRIMARY):
        sub = df[df["task"].eq(task)]

        sns.boxplot(
            data=sub,
            x="k_label",
            y="score",
            order=order,
            whis=(0, 100),
            showfliers=False,
            color="lightgray",
            width=0.55,
            ax=ax,
        )

        # One stripplot per model so each objective mix gets its own marker.
        for model, marker in MODEL_MARKERS.items():
            cells = sub[sub["model"].eq(model)]
            if cells.empty:
                continue

            sns.stripplot(
                data=cells,
                x="k_label",
                y="score",
                hue="dataset",
                order=order,
                hue_order=datasets,
                palette=palette,
                marker=marker,
                dodge=False,
                jitter=0.14,
                size=11 if marker == "*" else 7,
                edgecolor="white",
                linewidth=0.6,
                legend=False,
                ax=ax,
            )

        ax.set_title(TASK_TITLES[task])
        ax.set_xlabel("")
        ax.set_ylabel(METRIC_LABELS[PRIMARY[task]])

    dataset_handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="none",
            markerfacecolor=palette[dataset],
            markeredgecolor="white",
            markersize=9,
            label=dataset,
        )
        for dataset in datasets
    ]

    model_handles = [
        Line2D(
            [0], [0],
            marker=marker,
            linestyle="none",
            markerfacecolor="gray",
            markeredgecolor="white",
            markersize=12 if marker == "*" else 9,
            label=MODEL_LABELS[model],
        )
        for model, marker in MODEL_MARKERS.items()
    ]

    fig.legend(
        handles=dataset_handles,
        title="Dataset",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.88, 0.92),
    )

    fig.legend(
        handles=model_handles,
        title="Objective mix",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.88, 0.58),
    )

    fig.suptitle("Performance by number of pretraining objectives", y=1.02)

    sns.despine()
    fig.tight_layout(rect=[0, 0, 0.87, 1])
    return fig


if __name__ == "__main__":
    fig = plot(load())

    FIGURES.mkdir(exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(
            FIGURES / f"perf_by_k.{suffix}",
            dpi=200,
            bbox_inches="tight",
        )

    plt.show()
