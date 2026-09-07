"""Display paired representation/model effects, retaining every policy and seed."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


def main():
    root = Path(__file__).parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=root / "data")
    parser.add_argument("--output-dir", type=Path, default=root / "figures")
    args = parser.parse_args()
    receipt = json.loads((args.data_dir / "readout_intervention_validation.json").read_text())
    if receipt["rows"] != 8160 or receipt["upstream_tensor_comparisons"] != 153600:
        raise ValueError("complete verified swap analysis required")
    cells = pd.read_csv(args.data_dir / "readout_intervention_cells.csv")
    keys = ["stream", "intervention", "source", "policy", "seed", "dataset"]
    if len(cells) != 8160 or cells.duplicated(keys + ["decoder"]).any():
        raise ValueError("incomplete or duplicated swap analysis")
    pair = cells[cells.decoder.isin(["U1_pre_meta/ridge", "full_model"])].pivot(
        index=keys, columns="decoder", values="delta_roc_auc").reset_index()
    if len(pair) != 480 or pair.isna().any().any():
        raise ValueError("missing paired representation/model effect")
    colors = {"ukr_rus": "#226da4", "cp_hk": "#bb5824"}
    markers = {0: "o", 1: "s", 2: "^"}
    targets = [("facebook_page_reference", "Facebook pages"), ("twibot20", "TwiBot")]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for mode, title, filename, reference in (
        ("trained_background_initial_readout", "Restoring the initial readout is not a general model rescue",
         "readout_restoration_tradeoffs", "trained model"),
        ("initial_background_trained_readout", "Implanting the trained readout depends on the target and source",
         "readout_implant_tradeoffs", "exact initial model"),
    ):
        fig, axes = plt.subplots(2, 2, figsize=(11.7, 9.0), sharex=True, sharey=True)
        fig.subplots_adjust(left=.09, right=.97, top=.81, bottom=.19, hspace=.3, wspace=.18)
        shown = pair[(pair.intervention == mode) & pair.dataset.isin(dict(targets))]
        xmin, xmax = 100 * shown["U1_pre_meta/ridge"].agg(["min", "max"])
        ymin, ymax = 100 * shown.full_model.agg(["min", "max"])
        xlo, xhi = min(xmin, 0) - 1.6, max(xmax, 0) + 1.6
        ylo, yhi = min(ymin, 0) - 2.0, max(ymax, 0) + 2.0
        for row, stream in enumerate(("original", "fresh")):
            for col, (target, target_name) in enumerate(targets):
                ax = axes[row, col]
                subset = shown[(shown.stream == stream) & (shown.dataset == target)]
                if len(subset) != 24:
                    raise ValueError("all 24 source/policy/seed cases required in each panel")
                ax.axhline(0, color="#8c9399", lw=.8)
                ax.axvline(0, color="#8c9399", lw=.8)
                ax.axvspan(0, xhi, ymin=(0-ylo)/(yhi-ylo), ymax=1, color="#f0f7f0", zorder=0)
                for source, color in colors.items():
                    for seed, marker in markers.items():
                        group = subset[(subset.source == source) & (subset.seed == seed)]
                        for standard in (False, True):
                            points = group[(group.policy == "lowest_sorted") == standard]
                            if len(points) != (1 if standard else 3):
                                raise ValueError("incorrect policy coverage")
                            ax.scatter(100 * points["U1_pre_meta/ridge"], 100 * points.full_model,
                                       marker=marker, s=65 if standard else 36,
                                       facecolors=color if standard else "none", edgecolors=color,
                                       alpha=1 if standard else .45, linewidths=1.0,
                                       zorder=4 if standard else 3)
                ax.set_title(f"{target_name} · {stream} episodes", loc="left", fontsize=11)
                ax.set_xlim(xlo, xhi)
                ax.set_ylim(ylo, yhi)
                ax.spines[["top", "right"]].set_visible(False)
                ax.grid(color="#edf0f2", alpha=.7, zorder=0)
                if col == 0:
                    ax.set_ylabel("Full-model ΔAUC (percentage points)")
                if row == 1:
                    ax.set_xlabel("Readout-probe ΔAUC (percentage points)")
        fig.suptitle(title, fontsize=16, y=.965)
        fig.text(.09, .91, f"Only four readout tensors change; all other weights and inputs match the {reference}.", fontsize=10)
        handles = [Line2D([], [], marker="o", ls="", color=color, label=label)
                   for (source, color), label in zip(colors.items(), ("Ukraine", "Hong Kong"))]
        handles += [Line2D([], [], marker=marker, ls="", color="#505860", label=f"Seed {seed}")
                    for seed, marker in markers.items()]
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.08, .9),
                   ncol=5, frameon=False, handletextpad=.4, columnspacing=1.8)
        fig.text(.09, .07,
                 "Filled: original lowest-ID/sorted policy. Hollow: all three alternative policies; no models omitted.\n"
                 "Upper-right means both improve. A probe gain alone is not a full-model gain. Each point is one model,\n"
                 "not an independent domain or confidence interval. Both swap directions and all five targets are retained in the data.",
                 fontsize=9, linespacing=1.5)
        for suffix in ("png", "pdf"):
            fig.savefig(args.output_dir / f"{filename}.{suffix}", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()
