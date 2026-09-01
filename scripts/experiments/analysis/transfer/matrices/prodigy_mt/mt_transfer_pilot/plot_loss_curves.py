#!/usr/bin/env python3
"""Recover and plot pilot training losses from offline W&B records."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[7]
WANDB = ROOT / "wandb"
OUT = Path(__file__).parent
NAME_RE = re.compile(r"  exp_name: (mtpilot_(MT|NM_MT)_(.+)_\d{2}_\d{2}_\d{4}_.*)")
POINT_RE = re.compile(r"(?:\||\s)(\d+)/900 .*?loss=([0-9.]+)")
SHORT = {
    "covid_political": "COVID",
    "election2020": "Election",
    "facebook_page_reference": "Facebook",
    "twibot20": "TwiBot",
    "ukr_rus_suspended": "UKR-RUS",
}


rows = []
for run_dir in sorted(WANDB.glob("offline-run-20260831_17*")):
    config = run_dir / "files" / "effective_config.yaml"
    records = list(run_dir.glob("run-*.wandb"))
    if not config.exists() or not records:
        continue
    match = NAME_RE.search(config.read_text())
    if not match:
        continue
    _, arm, source = match.groups()
    if source not in SHORT:
        continue
    text = subprocess.run(["strings", str(records[0])], capture_output=True, text=True, check=True).stdout
    # TQDM/W&B can retain multiple renderings of a step; the last is the final rendering.
    points = {int(step): float(loss) for step, loss in POINT_RE.findall(text)}
    rows.extend({"arm": arm, "source": source, "step": step, "loss": loss}
                for step, loss in sorted(points.items()))

data = pd.DataFrame(rows)
if data.empty:
    raise SystemExit("No pilot loss histories found")
(OUT / "data").mkdir(exist_ok=True)
(OUT / "figures").mkdir(exist_ok=True)
data.to_csv(OUT / "data" / "training_loss_curves.csv", index=False)

fig, axes = plt.subplots(2, 5, figsize=(15, 5.5), sharex=True, sharey=True, constrained_layout=True)
for row, arm in enumerate(("MT", "NM_MT")):
    for col, source in enumerate(SHORT):
        ax = axes[row, col]
        curve = data[(data.arm == arm) & (data.source == source)].sort_values("step")
        ax.plot(curve.step, curve.loss, color="#3366cc" if arm == "MT" else "#dd4477", alpha=.25, lw=.7)
        ax.plot(curve.step, curve.loss.rolling(35, min_periods=5, center=True).median(),
                color="#174ea6" if arm == "MT" else "#b3144a", lw=1.8)
        ax.set_title(SHORT[source])
        if col == 0:
            ax.set_ylabel(f"{arm.replace('_', '+')}\ntraining loss")
        if row == 1:
            ax.set_xlabel("optimizer update")
        ax.grid(alpha=.2)
fig.suptitle("MT transfer pilot — raw batch loss and 35-step rolling median", fontsize=14)
fig.savefig(OUT / "figures" / "training_loss_curves.png", dpi=200)
fig.savefig(OUT / "figures" / "training_loss_curves.pdf")
print(f"plotted {len(data)} logged points across {data.groupby(['arm','source']).ngroups} runs")
