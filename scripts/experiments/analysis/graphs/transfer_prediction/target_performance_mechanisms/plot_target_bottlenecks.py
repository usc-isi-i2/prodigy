"""Scientific summary of fixed-support diagnostics on two episode streams."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    here = Path(__file__).parent
    data = here / "data"
    old, fresh = pd.read_csv(data / "replay_cells.csv"), pd.read_csv(data / "fresh_replay_cells.csv")
    scalar = [pd.read_json(data / "input_scalar_probes.json"), pd.read_json(data / "fresh_input_scalar_probes.json")]
    metadata = pd.read_json(data / "metadata_probe.json")
    panels = [
        ("covid_political", "covid-political", [("Raw bio + ridge", "raw_center/ridge"), ("Raw context + ridge", "raw_context/ridge"), ("Best full specialist*", "full_model")]),
        ("facebook_page_reference", "facebook-page-reference", [("Raw bio + ridge", "raw_center/ridge"), ("Raw context + ridge", "raw_context/ridge"), ("Best full specialist*", "full_model")]),
        ("twibot20", "twibot20 (retweet graph)", [("Raw bio + ridge", "raw_center/ridge"), ("Sampled in-degree", "scalar_center_indegree"), ("Untrained graph encoder + ridge", "U1_pre_meta/ridge"), ("Best full specialist*", "full_model")]),
        ("election2020", "election2020-political", [("Raw bio + ridge", "raw_center/ridge"), ("Context coherence", "scalar_context_coherence"), ("Raw context + ridge", "raw_context/ridge"), ("Best full specialist*", "full_model")]),
        ("ukr_rus_suspended", "ukraine-suspended", [("Raw bio + ridge", "raw_center/ridge"), ("Omitted account metadata†", "metadata_values"), ("Best full specialist*", "full_model")]),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.7))
    colors = {"full_model": "#b45f06", "scalar_center_indegree": "#7552a3", "scalar_context_coherence": "#7552a3", "metadata_values": "#b34b68", "U1_pre_meta/ridge": "#608239"}
    values = []
    for ax, (target, title, methods) in zip(axes.flat, panels):
        for index, (label, method) in enumerate(methods):
            pair = []
            for stream, cells in enumerate((old, fresh)):
                if method.startswith("scalar_"):
                    value = scalar[stream].query("dataset == @target and probe == @method").roc_auc.item()
                elif method == "metadata_values":
                    wanted = "fresh_stage_cpu_20260906" if stream else "specialist_cpu_tail_20260906"
                    value = metadata[(metadata.probe == method) & metadata.replay_root.str.endswith(wanted)].roc_auc.item()
                elif method == "full_model":
                    value = cells[(cells.dataset == target) & (cells.variant == "baseline") & (cells.decoder == method) & (cells.model_id != "random_init")].roc_auc.max()
                else:
                    value = cells[(cells.dataset == target) & (cells.variant == "baseline") & (cells.decoder == method) & (cells.model_id == "random_init")].roc_auc.item()
                pair.append(value)
                values.append({"dataset": target, "method": method, "stream": "original" if stream == 0 else "fresh", "roc_auc": value})
            color = colors.get(method, "#167d8d")
            ax.plot(pair, [index, index], color=color, alpha=.6, lw=2)
            ax.scatter(pair[0], index, s=64, facecolor="white", edgecolor=color, linewidth=1.7, zorder=3)
            ax.scatter(pair[1], index, s=46, facecolor=color, edgecolor="white", linewidth=.4, zorder=4)
        ax.set_title(title, loc="left", weight="bold", fontsize=12, pad=13)
        ax.set_yticks(range(len(methods)), [x[0] for x in methods], fontsize=9)
        ax.set_ylim(len(methods) - .5, -.6)
        ax.set_xlim(.45, 1.02)
        ax.set_xticks([.5, .6, .7, .8, .9, 1.])
        ax.set_xlabel("AUC", fontsize=9)
        ax.axvline(.5, color="#9aa3a9", linestyle=":", linewidth=1)
        ax.grid(axis="x", alpha=.16)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(axis="both", length=0)
    note = axes.flat[-1]
    note.axis("off")
    note.text(0, .98, "What this does—and does not—show", fontsize=12, weight="bold", va="top")
    note.text(0, .85, "○ Original episodes    ● Fresh episodes\n\nEach stream: 128 binary 10-shot episodes.\nProbe fitting uses episode supports only.\n\n* Best of nine specialists, chosen per stream:\n  an oracle summary, not a selection method.\n\n† Metadata were not model inputs;\n  timing relative to suspension is unverified.\n\nNo training-seed or new-domain replication.\nAvailable signal ≠ proof of model reliance.", fontsize=10, va="top", linespacing=1.5)
    fig.suptitle("Different targets expose different bottlenecks", x=.035, ha="left", y=.99, fontsize=20, weight="bold")
    fig.text(.035, .945, "Frozen checkpoints, verified input alignment, support-only probes, and a second episode stream.", fontsize=11, color="#53626d")
    fig.subplots_adjust(left=.15, right=.98, top=.86, bottom=.07, wspace=.96, hspace=.63)
    output = here / "figures"
    output.mkdir(exist_ok=True)
    fig.savefig(output / "target_bottlenecks.png", dpi=170, facecolor="white", bbox_inches="tight")
    fig.savefig(output / "target_bottlenecks.pdf", facecolor="white", bbox_inches="tight")
    pd.DataFrame(values).to_csv(data / "target_bottlenecks.csv", index=False)


if __name__ == "__main__":
    main()
