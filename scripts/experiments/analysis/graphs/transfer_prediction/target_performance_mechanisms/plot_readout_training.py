"""Show every seed/source training contrast, including the failed primary."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import pandas as pd


def main():
    root = Path(__file__).parent
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data-dir", type=Path, default=root / "data")
    parser.add_argument("--output-dir", type=Path, default=root / "figures")
    args = parser.parse_args()
    receipt = json.loads((args.data_dir / "readout_training_validation.json").read_text())
    if receipt["rows"] != 3060 or receipt["paired_changes"] != 1530 or not receipt["all_cached_inputs_match"]:
        raise ValueError("complete verified training analysis required")
    changes = pd.read_csv(args.data_dir / "readout_training_changes.csv")
    keys = ["stream", "source", "seed", "dataset"]
    if len(changes) != 1530 or changes.duplicated(keys + ["decoder"]).any():
        raise ValueError("incomplete or duplicated training contrast")
    pair = changes[changes.decoder.isin(["U1_pre_meta/ridge", "full_model"])].pivot(
        index=keys, columns="decoder", values="delta_roc_auc").reset_index()
    if len(pair) != 90 or pair.isna().any().any():
        raise ValueError("all 90 paired representation/model effects required")
    targets = [("facebook_page_reference", "Facebook · primary"), ("twibot20", "TwiBot"),
               ("covid_political", "COVID Political"), ("election2020", "Election"),
               ("ukr_rus_suspended", "Suspension")]
    sources = {"ukr_rus": ("Ukraine", "#226da4"), "cp_hk": ("Hong Kong", "#bb5824"),
               "covid": ("COVID", "#38835d")}
    markers = {0: "o", 1: "s", 2: "^"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(2, 5, figsize=(16, 7.5))
    fig.subplots_adjust(left=.065, right=.985, top=.77, bottom=.23, hspace=.33, wspace=.36)
    for col, (target, name) in enumerate(targets):
        both = pair[pair.dataset == target]
        x = both["U1_pre_meta/ridge"] * 100
        y = both.full_model * 100
        xlim = (min(0, x.min()) - 1.0, max(0, x.max()) + 1.0)
        ylim = (min(0, y.min()) - .8, max(0, y.max()) + .8)
        for row, stream in enumerate(("original", "fresh")):
            ax = axes[row, col]
            subset = both[both.stream == stream]
            if len(subset) != 9 or set(map(tuple, subset[["source", "seed"]].to_numpy())) != {
                    (source, seed) for source in sources for seed in markers}:
                raise ValueError("missing source/seed in plot panel")
            ax.axhline(0, color="#808890", lw=.9)
            ax.axvline(0, color="#808890", lw=.9)
            for source, (_, color) in sources.items():
                for seed, marker in markers.items():
                    point = subset[(subset.source == source) & (subset.seed == seed)].iloc[0]
                    ax.scatter(100 * point["U1_pre_meta/ridge"], 100 * point.full_model,
                               c=color, marker=marker, s=52, edgecolors="white", linewidths=.6, zorder=3)
            ax.set(xlim=xlim, ylim=ylim)
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.yaxis.set_major_locator(MaxNLocator(4))
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(color="#edf0f2", lw=.6)
            if row == 0:
                ax.set_title(name, loc="left", fontsize=11)
            if col == 0:
                ax.set_ylabel(f"{stream.capitalize()} episodes\nFull-model ΔAUC (points)")
            if row == 1:
                ax.set_xlabel("Readout-probe ΔAUC (points)")
    fig.suptitle("Freezing the readout improves Facebook probe access, but is not a consistent model rescue",
                 fontsize=15, x=.065, ha="left", y=.97)
    fig.text(.065, .91,
             "Frozen − freely trained: 18 new models, 3 sources × 3 seeds × 2 conditions; 2,500 updates each.\n"
             "Each treatment pair starts identically and consumes identical full input tensors. All other weights adapt.",
             fontsize=10, linespacing=1.6, va="top")
    handles = [Line2D([], [], color=color, marker="o", ls="", label=name)
               for name, color in sources.values()]
    handles += [Line2D([], [], color="#555d65", marker=marker, ls="", label=f"Seed {seed}")
                for seed, marker in markers.items()]
    fig.legend(handles=handles, ncol=6, frameon=False, loc="upper left", bbox_to_anchor=(.057, .86))
    verdict = "met" if receipt["primary_consistently_positive"] else "FAILED"
    fig.text(.065, .095,
             f"Prespecified Facebook full-model consistency criterion: {verdict}. The probe improves in all 9 pairs on both streams.\n"
             "Upper-right means both improve; every source/seed is shown. Axis ranges differ by target, but match across streams.\n"
             "Two episode streams are not independent domains. Same-seed free controls differ from earlier runs; that audit is retained.",
             fontsize=9, linespacing=1.6)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(args.output_dir / f"readout_training_tradeoffs.{suffix}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
