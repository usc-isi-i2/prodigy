#!/usr/bin/env python3
"""Plot complete separated NM and MT losses from offline W&B histories."""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore

ROOT = Path(__file__).resolve().parents[7]
WANDB = ROOT / "wandb"
OUT = Path(__file__).parent
SOURCES = {
    "covid_political": "COVID",
    "election2020": "Election",
    "facebook_page_reference": "Facebook",
    "twibot20": "TwiBot",
    "ukr_rus_suspended": "UKR-RUS",
}
EXP = re.compile(r"  exp_name: (mtpilot_NM_MT_(.+)_\d{2}_\d{2}_\d{4}_.*)")


def history(record_path):
    store = DataStore()
    store.open_for_scan(str(record_path))
    while True:
        raw = store.scan_data()
        if raw is None:
            return
        record = wandb_internal_pb2.Record()
        record.ParseFromString(raw)
        if not record.history.item:
            continue
        row = {}
        for item in record.history.item:
            key = item.key or ".".join(item.nested_key)
            row[key] = json.loads(item.value_json)
        if {"_step", "train_loss", "train_acc"} <= row.keys():
            yield row

rows = []
for source in SOURCES:
    candidates = []
    for run_dir in WANDB.glob("offline-run-*"):
        config = run_dir / "files/effective_config.yaml"
        records = list(run_dir.glob("run-*.wandb"))
        if not config.exists() or not records:
            continue
        match = EXP.search(config.read_text())
        if match and match.group(2) == source and "smoke" not in match.group(1):
            candidates.append((run_dir.name, records[0]))
    if not candidates:
        raise SystemExit(f"missing corrected W&B history for {source}")
    for point in history(sorted(candidates)[-1][1]):
        step = int(point["_step"])
        rows.append({"source": source, "objective": "NM" if step % 2 == 0 else "MT",
                     "update": step + 1, "accuracy": point["train_acc"],
                     "loss": point["train_loss"]})

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
