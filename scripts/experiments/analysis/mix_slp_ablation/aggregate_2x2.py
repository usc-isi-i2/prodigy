#!/usr/bin/env python3
"""Rebuild the FINDINGS 2x2 tables from data/slp_ablation_2x2.csv."""
from pathlib import Path
import pandas as pd

df = pd.read_csv(Path(__file__).parent / "data" / "slp_ablation_2x2.csv")
df = df[df.split == "test"]
assert not df.duplicated(["model", "dataset", "condition"]).any()
order = ["none", "rewire", "permute", "both"]
for model, sub in df.groupby("model"):
    t = sub.pivot_table(index="dataset", columns="condition", values="roc_auc")[order]
    t.loc["mean"] = t.mean()
    print(f"\n== {model} (test ROC-AUC) ==")
    print(t.round(3).to_string())
