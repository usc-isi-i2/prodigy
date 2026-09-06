"""Show every corrected TwiBot trajectory without selecting favorable checkpoints."""
import argparse
from itertools import product
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from .analyze_corrected_sampler import SOURCES, STEPS

DECODERS = {"S0_conv_center/ridge": "Center branch + ridge",
            "S0_pool/ridge": "Pooled branch + ridge",
            "U1_pre_meta/ridge": "Learned readout + ridge", "full_model": "Full PRODIGY"}


def panel_data(cells):
    shown = cells[cells.dataset.eq("twibot20") & cells.decoder.isin(DECODERS)].copy()
    keys = ["stream", "source", "step", "decoder"]
    expected = set(product(("original", "fresh"), SOURCES, STEPS, DECODERS))
    if len(shown) != len(expected) or shown.duplicated(keys).any() or set(map(tuple, shown[keys].to_numpy())) != expected:
        raise ValueError("all sources, steps, decoders and streams required")
    if not np.isfinite(shown.roc_auc).all() or not shown.roc_auc.between(0, 1).all():
        raise ValueError("invalid plotted AUC")
    return shown


def main():
    root = Path(__file__).parent
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data", type=Path, default=root / "data")
    parser.add_argument("--output", type=Path, default=root / "figures")
    args = parser.parse_args()
    receipt = json.loads((args.data / "corrected_sampler_validation.json").read_text())
    if receipt.get("rows") != 6120 or receipt.get("all_cached_inputs_match") is not True:
        raise ValueError("complete corrected analysis required")
    shown = panel_data(pd.read_csv(args.data / "corrected_sampler_cells.csv"))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(2, 4, figsize=(15, 8.5), sharex=True, sharey=True)
    fig.subplots_adjust(left=.07, right=.98, bottom=.23, top=.83, hspace=.32, wspace=.17)
    colors = plt.get_cmap("tab10")(np.arange(len(SOURCES)))
    lo = min(.45, np.floor(shown.roc_auc.min() * 20) / 20)
    hi = np.ceil(shown.roc_auc.max() * 20) / 20
    for row, stream in enumerate(("original", "fresh")):
        for col, (decoder, title) in enumerate(DECODERS.items()):
            ax = axes[row, col]
            selected = shown[shown.stream.eq(stream) & shown.decoder.eq(decoder)]
            for source, color in zip(SOURCES, colors):
                series = selected[selected.source.eq(source)].set_index("step").loc[list(STEPS)]
                ax.plot(STEPS, series.roc_auc, color=color, lw=.8, alpha=.65, marker="o", ms=2)
            mean = selected.groupby("step").roc_auc.mean().loc[list(STEPS)]
            ax.plot(STEPS, mean, color="#17212b", lw=2.4, marker="o", ms=3, zorder=10)
            ax.axhline(.5, color="#999", ls="--", lw=.7)
            ax.set_xscale("log")
            ax.set_xticks(STEPS, [str(s) for s in STEPS])
            ax.set_ylim(lo, hi)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_title(title if row == 0 else "", loc="left", fontsize=11)
            if col == 0:
                ax.set_ylabel(f"{stream.capitalize()} episodes\nAUC")
            if row == 1:
                ax.set_xlabel("Training updates")
    fig.suptitle("The corrected sampler does not resolve TwiBot's training mismatch", fontsize=17, y=.97)
    fig.text(.07, .89, "Every source improves the pooled probe from 100 to 2,500 updates; every full model declines, on both streams.", fontsize=11)
    handles = [Line2D([], [], color="#888", lw=1, label="Each of nine source runs (including TwiBot)") ,
               Line2D([], [], color="#17212b", lw=2.4, label="Unweighted nine-source mean")]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(.062, .12), ncol=2, frameon=False)
    fig.text(.07, .045, "One training seed; four fixed saved checkpoints; identical test inputs across sources and updates. No checkpoint selection.\n"
             "Center and pooled branches are parallel, not serial stages. Probe accessibility is not an information-theoretic or causal decomposition.\n"
             "The corrected recipe jointly changes retention, role order and randomness; these curves do not isolate those changes.", fontsize=9, linespacing=1.6)
    args.output.mkdir(exist_ok=True, parents=True)
    for extension in ("png", "pdf"):
        fig.savefig(args.output / f"corrected_sampler_twibot_trajectories.{extension}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
