from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path


PLOT_DIR = Path(__file__).resolve().parent
CACHE_DIR = Path(tempfile.gettempdir()) / "prodigy-plot-cache"
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR / "xdg"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import pandas as pd
import seaborn as sns


DATA = """experiment	train	test_ukr	test_covid	test_midterm	held_out
ukr_cov	single_ukr	0.9497	0.9741	0.884	midterm
ukr_cov	single_covid	0.9245	0.9815	0.8837	midterm
ukr_cov	merged_proportional@match	0.9373	0.9778	0.8746	midterm
ukr_cov	merged_proportional@full	0.9433	0.9807	0.8755	midterm
ukr_cov	merged_within@match	0.9447	0.9811	0.8835	midterm
ukr_cov	merged_within@full	0.9472	0.9822	0.8847	midterm
cov_mid	single_midterm	0.7935	0.8792	0.926	ukr
cov_mid	single_covid	0.9257	0.9813	0.886	ukr
cov_mid	merged_naive@match	0.9249	0.981	0.8849	ukr
cov_mid	merged_naive@full	0.924	0.9827	0.8899	ukr
cov_mid	merged_within@match	0.9243	0.9814	0.8943	ukr
cov_mid	merged_within@full	0.9244	0.9828	0.8984	ukr
cov_mid	merged_within_balanced@match	0.9216	0.978	0.9231	ukr
cov_mid	merged_within_balanced@full	0.9212	0.9796	0.9291	ukr"""


LABEL_OFFSETS = {
    "ukr_cov|merged_proportional@match|test_ukr": (8, -14),
    "ukr_cov|merged_proportional@match|test_covid": (8, 10),
    "ukr_cov|merged_within@match|test_ukr": (-54, 12),
    "ukr_cov|merged_within@match|test_covid": (8, -16),
    "cov_mid|merged_naive@match|test_covid": (-60, 24),
    "cov_mid|merged_naive@match|test_midterm": (8, -16),
    "cov_mid|merged_within@match|test_covid": (8, -24),
    "cov_mid|merged_within@match|test_midterm": (8, -16),
    "cov_mid|merged_within_balanced@match|test_covid": (-78, 10),
    "cov_mid|merged_within_balanced@match|test_midterm": (8, -16),
}


def strategy_name(train_value: str) -> str:
    if train_value.startswith("single"):
        return "single"
    return train_value


def distribution_type(row: pd.Series) -> str:
    train = row["train"]
    experiment = row["experiment"]
    test_dataset = row["Test_Dataset"]

    if train.startswith("single"):
        domain = train.split("_", maxsplit=1)[1]
        return "In-Distribution" if test_dataset == f"test_{domain}" else "Out-of-Distribution"

    if experiment == "ukr_cov":
        in_dist = test_dataset in {"test_ukr", "test_covid"}
    elif experiment == "cov_mid":
        in_dist = test_dataset in {"test_covid", "test_midterm"}
    else:
        raise ValueError(f"Unknown experiment: {experiment}")

    return "In-Distribution" if in_dist else "Out-of-Distribution"


def build_scatter_frame() -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(DATA), sep="\t")
    melted = df.melt(
        id_vars=["experiment", "train", "held_out"],
        value_vars=["test_ukr", "test_covid", "test_midterm"],
        var_name="Test_Dataset",
        value_name="ROCAUC",
    )
    melted["Strategy"] = melted["train"].apply(strategy_name)
    melted["Distribution_Type"] = melted.apply(distribution_type, axis=1)

    baseline = (
        melted[melted["Strategy"] == "single"]
        .groupby(["experiment", "Test_Dataset"], as_index=False)["ROCAUC"]
        .mean()
        .rename(columns={"ROCAUC": "Baseline_ROCAUC"})
    )

    melted = melted.merge(baseline, on=["experiment", "Test_Dataset"], validate="many_to_one")
    if (melted["Baseline_ROCAUC"] >= 1).any():
        raise ValueError("Cannot normalize by remaining headroom when a baseline ROCAUC is >= 1.")
    melted["Headroom_Normalized_Diff"] = (
        melted["ROCAUC"] - melted["Baseline_ROCAUC"]
    ) / (1 - melted["Baseline_ROCAUC"])

    merged = melted[melted["Strategy"] != "single"].copy()
    merged["Base_Strategy"] = merged["Strategy"].str.split("@").str[0]
    merged["Match_Type"] = merged["Strategy"].str.split("@").str[1]
    merged = merged[merged["Match_Type"] == "match"].copy()

    point_keys = ["experiment", "train", "held_out", "Strategy", "Base_Strategy", "Match_Type"]
    in_dist = (
        merged[merged["Distribution_Type"] == "In-Distribution"]
        [point_keys + ["Test_Dataset", "Headroom_Normalized_Diff"]]
        .rename(
            columns={
                "Test_Dataset": "In_Test_Dataset",
                "Headroom_Normalized_Diff": "In-Distribution",
            }
        )
    )
    out_dist = (
        merged[merged["Distribution_Type"] == "Out-of-Distribution"]
        [point_keys + ["Test_Dataset", "Headroom_Normalized_Diff"]]
        .rename(
            columns={
                "Test_Dataset": "Out_Test_Dataset",
                "Headroom_Normalized_Diff": "Out-of-Distribution",
            }
        )
    )
    scatter = in_dist.merge(
        out_dist,
        on=point_keys,
        validate="many_to_one",
    )
    scatter["Base_Strategy"] = scatter["Strategy"].str.split("@").str[0]
    scatter["Match_Type"] = scatter["Strategy"].str.split("@").str[1]
    scatter["Point_ID"] = (
        scatter["experiment"] + "|" + scatter["Strategy"] + "|" + scatter["In_Test_Dataset"]
    )
    return scatter


def format_label(row: pd.Series) -> str:
    base = row["Base_Strategy"].replace("merged_", "").replace("_", " ")
    trained_model = row["experiment"].replace("_", "+")
    test_dataset = row["In_Test_Dataset"].replace("test_", "")
    return f"{base}\ntrain: {trained_model}\neval: {test_dataset}"


def padded_limits(values: pd.Series, extra: float = 0.16) -> tuple[float, float]:
    low = min(values.min(), 0)
    high = max(values.max(), 0)
    span = high - low
    pad = max(span * extra, 0.004)
    return low - pad, high + pad


def plot() -> tuple[Path, Path]:
    scatter = build_scatter_frame()

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(13.5, 8.4), layout="constrained")

    palette = {
        "merged_naive": "#4C78A8",
        "merged_proportional": "#F58518",
        "merged_within": "#54A24B",
        "merged_within_balanced": "#B279A2",
    }
    markers = {
        "test_covid": "o",
        "test_midterm": "s",
        "test_ukr": "^",
    }
    sns.scatterplot(
        data=scatter,
        x="In-Distribution",
        y="Out-of-Distribution",
        hue="Base_Strategy",
        style="In_Test_Dataset",
        palette=palette,
        markers=markers,
        s=50,
        edgecolor="#1f1f1f",
        linewidth=1.2,
        ax=ax,
        zorder=3,
    )

    ax.axhline(0, color="#2b2b2b", linewidth=1.15, linestyle=(0, (5, 4)), zorder=1)
    ax.axvline(0, color="#2b2b2b", linewidth=1.15, linestyle=(0, (5, 4)), zorder=1)
    ax.set_xlim(*padded_limits(scatter["In-Distribution"]))
    ax.set_ylim(*padded_limits(scatter["Out-of-Distribution"]))

    for _, row in scatter.iterrows():
        dx, dy = LABEL_OFFSETS[row["Point_ID"]]
        ax.annotate(
            format_label(row),
            xy=(row["In-Distribution"], row["Out-of-Distribution"]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="left" if dx >= 0 else "right",
            va="bottom" if dy >= 0 else "top",
            fontsize=9.3,
            fontweight="semibold",
            linespacing=1.15,
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "#d2d2d2",
                "linewidth": 0.8,
                "alpha": 0.94,
            },
            arrowprops={
                "arrowstyle": "-",
                "color": "#8a8a8a",
                "linewidth": 0.8,
                "shrinkA": 0,
                "shrinkB": 6,
            },
            zorder=4,
        )

    ax.annotate(
        "single baseline\n(0, 0)",
        xy=(0, 0),
        xytext=(-92, 14),
        textcoords="offset points",
        fontsize=9.5,
        ha="right",
        va="bottom",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "#cfcfcf",
            "alpha": 0.94,
        },
        zorder=4,
    )

    ax.text(
        0.99,
        0.03,
        "Headroom-normalized change = (model ROCAUC - baseline ROCAUC) / (1 - baseline ROCAUC).\n"
        "It measures the share of remaining possible improvement; 0.90 -> 0.95 is +50%.\n"
        "Each point is one @match run evaluated on one in-distribution dataset, paired with its held-out dataset.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.2,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#d0d0d0",
            "alpha": 0.95,
        },
        zorder=4,
    )

    ax.set_title("Strategy Trade-off: Headroom-Normalized ROCAUC Change", pad=16)
    ax.set_xlabel("In-distribution change (% of remaining headroom)")
    ax.set_ylabel("Out-of-distribution change (% of remaining headroom)")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(True, color="#e6e6e6", linewidth=0.9)
    ax.set_axisbelow(True)

    if ax.get_legend() is not None:
        ax.get_legend().remove()

    strategy_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            label=name.replace("merged_", "").replace("_", " "),
            markerfacecolor=color,
            markeredgecolor="#1f1f1f",
            markersize=9,
        )
        for name, color in palette.items()
    ]
    marker_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            color="none",
            label=dataset.replace("test_", ""),
            markerfacecolor="#777777",
            markeredgecolor="#1f1f1f",
            markersize=9,
        )
        for dataset, marker in markers.items()
    ]
    legend = fig.legend(
        handles=[*strategy_handles, *marker_handles],
        title="Color = strategy, marker = in-distribution eval dataset",
        loc="outside lower center",
        ncol=4,
        frameon=True,
        fontsize=9.5,
        title_fontsize=10,
    )
    legend.get_frame().set_edgecolor("#d0d0d0")
    legend.get_frame().set_linewidth(0.8)

    png_path = PLOT_DIR / "strategy_tradeoff_headroom_normalized_roauc.png"
    pdf_path = PLOT_DIR / "strategy_tradeoff_headroom_normalized_roauc.pdf"
    fig.savefig(png_path, dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path, pdf_path


if __name__ == "__main__":
    png, pdf = plot()
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")
