#!/usr/bin/env python3
"""Plot separated NM and MT losses from the corrected alternating-task run logs."""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[7]
LOGS = ROOT / "scripts/experiments/setup/mt_transfer_pilot/run_logs"
OUT = Path(__file__).parent
SOURCES = {
    "covid_political": "COVID",
    "election2020": "Election",
    "facebook_page_reference": "Facebook",
    "twibot20": "TwiBot",
    "ukr_rus_suspended": "UKR-RUS",
}
POINT = re.compile(r"(\d+)/900.*?acc=([0-9.]+).*?loss=([0-9.]+)")

rows = []
for source in SOURCES:
    candidates = sorted(LOGS.glob(f"full_NM_MT_{source}_gpu*_*.log"))
    if not candidates:
        raise SystemExit(f"missing corrected log for {source}")
    matches = POINT.findall(candidates[-1].read_text(errors="ignore").replace("\r", "\n"))
    by_display_step = {}
    for step, acc, loss in matches:
        point = (float(acc), float(loss))
        bucket = by_display_step.setdefault(int(step), [])
        if not bucket or point != bucket[-1]:
            bucket.append(point)
    # TQDM refreshes after pairs of updates. At each odd displayed counter, the
    # first distinct metric is NM and the second is MT (step zero is the initial NM).
    for step, points in sorted(by_display_step.items()):
        if step == 0 and points:
            pairs = [("NM", points[0])]
        elif step % 2 == 1 and len(points) >= 2:
            pairs = [("NM", points[0]), ("MT", points[1])]
        else:
            continue
        for objective, (acc, loss) in pairs:
            rows.append({"source": source, "objective": objective,
                         "update": step, "accuracy": acc, "loss": loss})

data = pd.DataFrame(rows)
data.to_csv(OUT / "data/corrected_nm30_training_curves.csv", index=False)
fig, axes = plt.subplots(2, 5, figsize=(15, 5.6), sharex=True, constrained_layout=True)
for row, objective in enumerate(("NM", "MT")):
    for col, (source, short) in enumerate(SOURCES.items()):
        ax = axes[row, col]
        curve = data[(data.source == source) & (data.objective == objective)]
        color = "#174ea6" if objective == "NM" else "#b3144a"
        ax.plot(curve["update"], curve.loss, color=color, alpha=.16, lw=.6)
        ax.plot(curve["update"], curve.loss.rolling(25, min_periods=5, center=True).median(),
                color=color, lw=2)
        ax.set_title(short)
        ax.grid(alpha=.2)
        if col == 0:
            ax.set_ylabel(f"{objective} loss")
        if row == 1:
            ax.set_xlabel("total optimizer update")
        if objective == "NM":
            ax.axhline(3.401, color="black", ls="--", lw=.8, alpha=.45)
fig.suptitle("Corrected NM+MT: alternating 30-way NM and 2-way MT batches", fontsize=14)
fig.savefig(OUT / "figures/corrected_nm30_training_curves.png", dpi=200)
fig.savefig(OUT / "figures/corrected_nm30_training_curves.pdf")
print(data.groupby(["objective", "source"]).size())
