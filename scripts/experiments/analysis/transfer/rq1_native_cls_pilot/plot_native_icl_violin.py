#!/usr/bin/env python3
"""Plot seed-level native PRODIGY ICL results as violins with raw points."""

import json
import re
import shlex
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REMOTE_ROOT = "/dataMeR1/phil/gfm/prodigy-rq1-native/log/rq1_native_icl_v1/eval"
TARGETS = [
    ("covid_political", "COVID-19"),
    ("election2020", "Election 2020"),
    ("ukr_rus_suspended", "Ukraine/Russia"),
    ("twibot20", "TwiBot-20"),
]
SHOTS = [1, 3, 5, 10]
ARMS = [("pretrained", "Pretrained PRODIGY", "#1f77b4"),
        ("no_pretrain", "Random NoPretrain", "#ff7f0e")]


def load_results():
    remote_code = r'''import json,pathlib,re
root=pathlib.Path("''' + REMOTE_ROOT + r'''")
out=[]
for p in root.glob("rq1_icl_*/data/metrics_test_step0.json"):
    m=re.match(r"rq1_icl_(covid_political|election2020|ukr_rus_suspended|twibot20)_(1|3|5|10)shot_ls([0-4])_(pretrained|no_pretrain)_20260830", p.parent.parent.name)
    if m:
        d=json.loads(p.read_text())
        out.append([m.group(1),int(m.group(2)),int(m.group(3)),m.group(4),d["test_roc_auc"]])
print(json.dumps(out))'''
    remote_command = "/home/mhchu/miniconda3/bin/python -c " + shlex.quote(remote_code)
    raw = subprocess.check_output([
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "tucker",
        remote_command,
    ], text=True)
    rows = json.loads(raw)
    if len(rows) != 160:
        raise RuntimeError(f"expected 160 complete cells, found {len(rows)}")
    return rows


def main():
    rows = load_results()
    values = {}
    for target, shot, seed, arm, auc in rows:
        values.setdefault((target, shot, arm), []).append((seed, auc))

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    rng = np.random.default_rng(20260830)
    offsets = {"pretrained": -0.18, "no_pretrain": 0.18}

    for ax, (target, title) in zip(axes.flat, TARGETS):
        for arm, label, color in ARMS:
            groups = []
            positions = []
            for x, shot in enumerate(SHOTS, start=1):
                pairs = sorted(values[(target, shot, arm)])
                if [s for s, _ in pairs] != list(range(5)):
                    raise RuntimeError(f"incomplete seeds for {(target, shot, arm)}")
                ys = np.asarray([v for _, v in pairs])
                pos = x + offsets[arm]
                groups.append(ys)
                positions.append(pos)
                jitter = rng.uniform(-0.035, 0.035, size=len(ys))
                ax.scatter(pos + jitter, ys, s=28, color=color, edgecolor="white",
                           linewidth=0.55, zorder=3)
                ax.scatter(pos, ys.mean(), marker="D", s=42, color=color,
                           edgecolor="black", linewidth=0.55, zorder=4)
            parts = ax.violinplot(groups, positions=positions, widths=0.30,
                                  showmeans=False, showmedians=False, showextrema=False,
                                  bw_method=0.55)
            for body in parts["bodies"]:
                body.set_facecolor(color)
                body.set_edgecolor(color)
                body.set_alpha(0.28)
            ax.scatter([], [], s=45, color=color, label=label)

        ax.axhline(0.5, color="#999999", linestyle="--", linewidth=1.2, alpha=0.8)
        ax.set_title(title, fontsize=17)
        ax.set_xticks(range(1, 5), SHOTS)
        ax.set_xlabel("Prompt labels per class (shots)", fontsize=13)
        ax.set_ylabel("Test ROC-AUC", fontsize=13)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(labelsize=11)

    axes[0, 0].legend(frameon=False, fontsize=12, loc="center right")
    fig.suptitle("Zero-update native PRODIGY in-context transfer\n"
                 "Violin = distribution across 5 prompt seeds; dots = seeds; diamond = mean",
                 fontsize=20)
    out = Path(__file__).parent / "figures" / "rq1_native_icl_shots_violin.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
