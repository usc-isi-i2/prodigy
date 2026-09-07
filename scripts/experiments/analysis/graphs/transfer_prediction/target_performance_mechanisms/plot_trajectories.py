"""Show all donors' prespecified endpoint changes, not oracle checkpoints."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    root = Path(__file__).parent
    data = pd.read_csv(root / "data/trajectory_endpoint_changes.csv")
    sources = ["covid", "ukr_rus", "twibot20", "midterm", "covid_political",
               "cp_hk", "election2020", "ukr_rus_suspended", "facebook_page_reference"]
    source_names = ["COVID", "Ukraine", "TwiBot20", "Midterm", "COVID Political",
                    "Hong Kong", "Election2020", "Ukraine Suspended", "Facebook"]
    targets = ["covid_political", "facebook_page_reference", "twibot20", "election2020", "ukr_rus_suspended"]
    target_names = ["COVID\nPolitical", "Facebook", "TwiBot20", "Election2020", "Ukraine\nSuspended"]
    decoders = {"S0_pool/ridge": "Pooled encoder + ridge", "U1_pre_meta/ridge": "Learned readout + ridge", "full_model": "Full PRODIGY"}
    selected = data[data.decoder.isin(decoders)]
    bound = max(.1, np.ceil(selected.change_100_to_2500.abs().max() * 10) / 10)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), layout="constrained")
    for row, (decoder, name) in enumerate(decoders.items()):
        for col, stream in enumerate(["original", "fresh"]):
            ax = axes[row, col]
            values = data[(data.decoder == decoder) & (data.stream == stream)].pivot(
                index="source", columns="dataset", values="change_100_to_2500").loc[sources, targets].to_numpy()
            if not np.isfinite(values).all():
                raise ValueError("incomplete heatmap")
            img = ax.imshow(values, cmap="RdBu", vmin=-bound, vmax=bound, aspect="auto")
            for i, j in np.ndindex(values.shape):
                ax.text(j, i, f"{values[i,j]:+.3f}", ha="center", va="center", fontsize=9,
                        color="white" if abs(values[i,j]) > .58 * bound else "#17212b")
            ax.set_xticks(range(len(targets)), target_names)
            ax.set_yticks(range(len(sources)), source_names)
            ax.set_title(f"{name} · {stream} episodes", fontsize=12, loc="left", pad=10)
            ax.set_xticks(np.arange(-.5, len(targets), 1), minor=True)
            ax.set_yticks(np.arange(-.5, len(sources), 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=1.5)
            ax.tick_params(which="both", length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
    fig.suptitle("What changes during training?\nAUC at 2,500 updates minus AUC at 100 updates", fontsize=18)
    fig.colorbar(img, ax=axes, shrink=.65, label="Endpoint ΔAUC · blue = increase, red = decrease")
    fig.supxlabel("One historical training seed; fixed inputs within each stream. Includes source=target cells.\nCommon-decoder changes are not a causal decomposition or proof of information loss.", fontsize=10)
    (root / "figures").mkdir(exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(root / f"figures/training_stage_changes.{extension}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
