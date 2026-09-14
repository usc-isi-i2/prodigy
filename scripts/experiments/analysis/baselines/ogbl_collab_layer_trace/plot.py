#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--results", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
r = json.loads(args.results.read_text())

fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
labels = ["Raw", "SAGE 1", "SAGE 2", "SAGE 3"]
keys = ["raw_node_features", "layer1", "layer2", "layer3"]
auc = [100 * r["endpoint_cosine"][key]["auc"] for key in keys]
hits = [100 * r["endpoint_cosine"][key]["hits_at_50"] for key in keys]
axes[0].plot(labels, auc, marker="o", label="AUC")
axes[0].plot(labels, hits, marker="o", label="Hits@50")
axes[0].set(title="Endpoint cosine becomes less discriminative", ylabel="Metric (%)")
axes[0].legend(frameon=False)

decoder_labels = ["Layer 1", "Layer 2", "Layer 3", "No learned\npair input", "No structural\ninput"]
decoder_keys = ["layer1", "layer2", "layer3", "zero_learned_pair", "zero_structural"]
decoder_hits = [100 * r["decoder_scores"][key]["hits_at_50"] for key in decoder_keys]
colors = ["#7895b2", "#7895b2", "#d45d3a", "#8f8f8f", "#8f8f8f"]
axes[1].bar(decoder_labels, decoder_hits, color=colors)
axes[1].set(title="Frozen decoder interventions", ylabel="Validation Hits@50 (%)")
axes[1].tick_params(axis="x", labelrotation=0, labelsize=8)
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.2)
fig.suptitle("Frozen seed-0 representation trace (diagnostic only)")
args.out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.out, dpi=180)
